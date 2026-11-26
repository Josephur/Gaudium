//! Rust Shell wrappers for tensor alignment, channel permutation, kernel-baked dimension
//! validation, ViT patch grids, draft-tree bounds, and lane state-block relocation.
//!
//! All predicates and bounds math delegate directly to `verified/tensor_bounds.bend`.

use crate::verified::{
    align128, align_up, gdn_dims_valid, gqa_dims_valid, hidden_dims_valid, kv8_vpos, kv8_vpos128,
    lane128, lane256, moe_dims_valid, tree_node_valid, tree_shape_bounds_valid,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    InvalidGqa {
        n_heads: u64,
        n_kv_heads: u64,
        head_dim: u64,
    },
    InvalidGdn {
        k_heads: u64,
        v_heads: u64,
        conv_k: u64,
    },
    InvalidHidden {
        hidden: u64,
        vocab: u64,
    },
    InvalidMoe {
        n_experts: u64,
        top_k: u64,
        expert_ff: u64,
    },
    InvalidTreeShape {
        n_nodes: u64,
        spine: u64,
    },
    InvalidTreeNode {
        idx: u64,
        parent: u64,
        depth: u64,
        rank: u64,
    },
}

/// Safe upward alignment (div-by-zero safe: `align == 0` returns `x`).
pub fn align_up_safe(x: u64, align: u64) -> u64 {
    align_up(x, align)
}

/// 128-element / 128-byte upward alignment.
pub fn align_128(x: u64) -> u64 {
    align128(x)
}

/// Deinterleaved 128-lane channel permutation (`model_qwen35.hpp`).
pub fn permute_lane128(c: u64) -> u64 {
    lane128(c)
}

/// Deinterleaved 256-channel head permutation (`model_qwen35.hpp`).
pub fn permute_lane256(c: u64) -> u64 {
    lane256(c)
}

/// FP8 KV V-row permutation within a 128-row block (`model_qwen35.hpp`).
pub fn permute_kv8_vpos128(q: u64) -> u64 {
    kv8_vpos128(q)
}

/// FP8 KV V-row permutation across blocks (`model_qwen35.hpp`).
pub fn permute_kv8_vpos(q: u64) -> u64 {
    kv8_vpos(q)
}

/// Validates kernel-baked GQA attention dimensions (`adapter_qwen35.hpp`).
pub fn validate_gqa(n_heads: u64, n_kv_heads: u64, head_dim: u64) -> Result<(), ShapeError> {
    if gqa_dims_valid(n_heads, n_kv_heads, head_dim) {
        Ok(())
    } else {
        Err(ShapeError::InvalidGqa {
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }
}

/// Validates kernel-baked Gated DeltaNet dimensions (`adapter_qwen35.hpp`).
pub fn validate_gdn(k_heads: u64, v_heads: u64, conv_k: u64) -> Result<(), ShapeError> {
    if gdn_dims_valid(k_heads, v_heads, conv_k) {
        Ok(())
    } else {
        Err(ShapeError::InvalidGdn {
            k_heads,
            v_heads,
            conv_k,
        })
    }
}

/// Validates hidden & vocabulary alignment bounds (`adapter_qwen35.hpp`).
pub fn validate_hidden(hidden: u64, vocab: u64) -> Result<(), ShapeError> {
    if hidden_dims_valid(hidden, vocab) {
        Ok(())
    } else {
        Err(ShapeError::InvalidHidden { hidden, vocab })
    }
}

/// Validates MoE expert counts and expert FF alignment (`adapter_qwen35.hpp`).
pub fn validate_moe(n_experts: u64, top_k: u64, expert_ff: u64) -> Result<(), ShapeError> {
    if moe_dims_valid(n_experts, top_k, expert_ff) {
        Ok(())
    } else {
        Err(ShapeError::InvalidMoe {
            n_experts,
            top_k,
            expert_ff,
        })
    }
}

/// Draft-tree node metadata matching `dtree::Node`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNodeSpec {
    pub parent: u64,
    pub depth: u64,
    pub rank: u64,
}

/// Validates a speculative caterpillar draft tree against verified node/depth/bitmask bounds.
pub fn validate_draft_tree(
    nodes: &[TreeNodeSpec],
    spine: u64,
    max_nodes: u64,
    max_depth: u64,
    topk: u64,
) -> Result<(), ShapeError> {
    let n_nodes = nodes.len() as u64;
    if !tree_shape_bounds_valid(n_nodes, spine, max_nodes, max_depth, topk) {
        return Err(ShapeError::InvalidTreeShape { n_nodes, spine });
    }
    // Root node 0 must be at depth 0, rank 0
    if nodes[0].depth != 0 || nodes[0].rank != 0 {
        return Err(ShapeError::InvalidTreeNode {
            idx: 0,
            parent: nodes[0].parent,
            depth: nodes[0].depth,
            rank: nodes[0].rank,
        });
    }
    for (idx, node) in nodes.iter().enumerate().skip(1) {
        let parent_idx = node.parent as usize;
        if parent_idx >= idx {
            return Err(ShapeError::InvalidTreeNode {
                idx: idx as u64,
                parent: node.parent,
                depth: node.depth,
                rank: node.rank,
            });
        }
        let parent_depth = nodes[parent_idx].depth;
        if !tree_node_valid(
            idx as u64,
            node.parent,
            node.depth,
            parent_depth,
            node.rank,
            spine,
            topk,
        ) {
            return Err(ShapeError::InvalidTreeNode {
                idx: idx as u64,
                parent: node.parent,
                depth: node.depth,
                rank: node.rank,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_bounds() {
        assert_eq!(align_up_safe(0, 128), 0);
        assert_eq!(align_up_safe(1, 128), 128);
        assert_eq!(align_up_safe(128, 128), 128);
        assert_eq!(align_up_safe(129, 128), 256);
        assert_eq!(align_up_safe(777, 0), 777); // div-by-zero safe
        assert_eq!(align_128(5120), 5120);
    }

    #[test]
    fn channel_and_kv8_permutations_are_bijections() {
        let mut seen_l128 = [false; 128];
        for c in 0..128u64 {
            let p = permute_lane128(c) as usize;
            assert!(
                p < 128 && !seen_l128[p],
                "lane128 collision at c={c} -> {p}"
            );
            seen_l128[p] = true;
        }
        assert_eq!(permute_lane128(0), 0);
        assert_eq!(permute_lane128(1), 64);
        assert_eq!(permute_lane128(2), 1);
        assert_eq!(permute_lane128(127), 127);

        let mut seen_l256 = [false; 256];
        for c in 0..256u64 {
            let p = permute_lane256(c) as usize;
            assert!(
                p < 256 && !seen_l256[p],
                "lane256 collision at c={c} -> {p}"
            );
            seen_l256[p] = true;
        }

        let mut seen_v128 = [false; 128];
        for q in 0..128u64 {
            let p = permute_kv8_vpos128(q) as usize;
            assert!(
                p < 128 && !seen_v128[p],
                "kv8_vpos128 collision at q={q} -> {p}"
            );
            seen_v128[p] = true;
        }
        assert_eq!(permute_kv8_vpos(128), 128);
        assert_eq!(permute_kv8_vpos(130), 192);
    }

    #[test]
    fn real_gaudi_model_dimensions_validate() {
        // Qwen3.8-27B (dense hybrid GQA + GDN)
        validate_gqa(24, 4, 256).unwrap();
        validate_gdn(16, 48, 4).unwrap();
        validate_hidden(5120, 248_320).unwrap();

        // Qwen3.6-35B-A3B (MoE hybrid)
        validate_gqa(16, 2, 256).unwrap();
        validate_gdn(16, 32, 4).unwrap();
        validate_moe(256, 8, 512).unwrap();

        // Qwen3-8B (pure GQA, dh=128)
        validate_gqa(32, 8, 128).unwrap();
        validate_hidden(4096, 151_936).unwrap();

        // Zero-dim and misalignment guards
        assert!(validate_gqa(24, 0, 256).is_err());
        assert!(validate_gqa(24, 5, 256).is_err());
        assert!(validate_gqa(24, 4, 64).is_err());
        assert!(validate_gdn(0, 48, 4).is_err());
        assert!(validate_gdn(16, 0, 4).is_err());
        assert!(validate_gdn(16, 48, 3).is_err());
        assert!(validate_hidden(0, 248_320).is_err());
        assert!(validate_hidden(5120, 248_321).is_err());
        assert!(validate_moe(0, 8, 512).is_err());
        assert!(validate_moe(256, 16, 512).is_err()); // ek > 12
    }

    #[test]
    fn draft_tree_validation_mirrors_tree_manager_test() {
        // Shape "4:1-4@2": 5 spine nodes (0..=4) + 4 leaves (5..=8)
        let mut t = vec![
            TreeNodeSpec {
                parent: 0,
                depth: 0,
                rank: 0,
            },
            TreeNodeSpec {
                parent: 0,
                depth: 1,
                rank: 0,
            },
            TreeNodeSpec {
                parent: 1,
                depth: 2,
                rank: 0,
            },
            TreeNodeSpec {
                parent: 2,
                depth: 3,
                rank: 0,
            },
            TreeNodeSpec {
                parent: 3,
                depth: 4,
                rank: 0,
            },
            TreeNodeSpec {
                parent: 0,
                depth: 1,
                rank: 1,
            },
            TreeNodeSpec {
                parent: 1,
                depth: 2,
                rank: 1,
            },
            TreeNodeSpec {
                parent: 2,
                depth: 3,
                rank: 1,
            },
            TreeNodeSpec {
                parent: 3,
                depth: 4,
                rank: 1,
            },
        ];
        validate_draft_tree(&t, 4, 32, 8, 4).unwrap();

        // Child of a leaf (parent == 5 >= spine 4) is rejected
        t.push(TreeNodeSpec {
            parent: 5,
            depth: 2,
            rank: 0,
        });
        assert!(validate_draft_tree(&t, 4, 32, 8, 4).is_err());

        // Self-parent is rejected
        let cyclic = vec![
            TreeNodeSpec {
                parent: 0,
                depth: 0,
                rank: 0,
            },
            TreeNodeSpec {
                parent: 1,
                depth: 1,
                rank: 0,
            },
        ];
        assert!(validate_draft_tree(&cyclic, 1, 32, 8, 4).is_err());

        // Exceeding adapter max_nodes (32 for u32 ancestor bitmask) is rejected
        let mut oversized = vec![TreeNodeSpec {
            parent: 0,
            depth: 0,
            rank: 0,
        }];
        for i in 1..33u64 {
            oversized.push(TreeNodeSpec {
                parent: i - 1,
                depth: i,
                rank: 0,
            });
        }
        assert!(validate_draft_tree(&oversized, 32, 32, 32, 4).is_err());
    }
}
