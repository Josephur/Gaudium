//! Model-agnostic speculative draft-tree manager (`tree_manager.hpp`).
//!
//! Parses static caterpillar specs (`"4"`, `"4:1-4@2"`, `"5:1-2@2-3,3-5@2"`) and device-planned
//! specs (`"d9@4"`), validates structural bounds via `core::shapes::validate_draft_tree`, builds
//! 32-bit ancestor bitmask tables, walks verified target paths, and runs online EMA acceptance
//! + latency shape selection.

use std::collections::BTreeMap;
use std::fmt;

use crate::core::shapes::{validate_draft_tree, TreeNodeSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    pub parent: i32,
    pub depth: usize,
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    Parse(String),
    Invalid(String),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(m) => write!(f, "tree spec parse error: {m}"),
            Self::Invalid(m) => write!(f, "invalid draft tree: {m}"),
        }
    }
}

impl std::error::Error for TreeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeShape {
    pub name: String,
    pub nodes: Vec<Node>,
    pub dynamic: bool,
    pub dyn_rows: usize,
    pub dyn_depth: usize,
}

impl TreeShape {
    pub fn n_nodes(&self) -> usize {
        if self.dynamic {
            self.dyn_rows
        } else {
            self.nodes.len()
        }
    }

    pub fn spine(&self) -> usize {
        if self.dynamic {
            return self.dyn_depth;
        }
        let mut d = 0usize;
        for t in 1..self.nodes.len() {
            if self.nodes[t].parent == (t as i32) - 1 && self.nodes[t].rank == 0 {
                d = t;
            } else {
                break;
            }
        }
        d
    }

    pub fn max_depth(&self) -> usize {
        if self.dynamic {
            self.dyn_depth
        } else {
            self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
        }
    }

    pub fn leaves(&self) -> usize {
        self.n_nodes().saturating_sub(1 + self.spine())
    }

    pub fn parent_tab(&self) -> Vec<i32> {
        let mut v: Vec<i32> = self.nodes.iter().map(|n| n.parent).collect();
        if !v.is_empty() {
            v[0] = -1;
        }
        v
    }

    pub fn depth_tab(&self) -> Vec<i32> {
        self.nodes.iter().map(|n| n.depth as i32).collect()
    }

    pub fn rank_tab(&self) -> Vec<i32> {
        self.nodes.iter().map(|n| n.rank as i32).collect()
    }

    /// 32-bit ancestor bitmask table: bit `j` of `anc[t]` is 1 iff `j` is `t` or an ancestor of `t`.
    pub fn anc_tab(&self) -> Vec<u32> {
        let mut v = vec![0u32; self.nodes.len()];
        for t in 0..self.nodes.len() {
            let mut u = t as i32;
            while u >= 0 {
                v[t] |= 1u32 << (u as u32);
                u = self.nodes[u as usize].parent;
            }
        }
        v
    }

    /// Generic tree validity check (DAG ordering, <= 32 nodes, no duplicate `(parent, rank)`).
    pub fn validate(&self) -> Result<(), TreeError> {
        if self.dynamic {
            if self.dyn_rows < 2 || self.dyn_rows > 32 || self.dyn_depth < 1 {
                return Err(TreeError::Invalid(format!(
                    "planned shape {}: rows 2..32, depth >= 1",
                    self.name
                )));
            }
            return Ok(());
        }
        if self.nodes.is_empty() || self.nodes[0].parent != -1 || self.nodes[0].depth != 0 {
            return Err(TreeError::Invalid("node 0 must be the root".into()));
        }
        if self.nodes.len() > 32 {
            return Err(TreeError::Invalid(
                "more than 32 nodes (ancestor bitmask)".into(),
            ));
        }
        for t in 1..self.nodes.len() {
            let n = self.nodes[t];
            if n.parent < 0 || (n.parent as usize) >= t {
                return Err(TreeError::Invalid(format!(
                    "node {t}: parent {} must precede it",
                    n.parent
                )));
            }
            if n.depth != self.nodes[n.parent as usize].depth + 1 {
                return Err(TreeError::Invalid(format!(
                    "node {t}: depth != parent depth + 1"
                )));
            }
            for u in 1..t {
                if self.nodes[u].parent == n.parent && self.nodes[u].rank == n.rank {
                    return Err(TreeError::Invalid(format!(
                        "node {t}: duplicate (parent, rank)"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validates caterpillar adapter bounds against the Bend 2 verified `validate_draft_tree`.
    pub fn validate_caterpillar(
        &self,
        max_nodes: u64,
        max_depth: u64,
        topk: u64,
    ) -> Result<(), TreeError> {
        self.validate()?;
        if self.dynamic {
            let n = self.dyn_rows as u64;
            let d = self.dyn_depth as u64;
            if n > max_nodes || d > max_depth {
                return Err(TreeError::Invalid(format!(
                    "planned shape {} exceeds max_nodes={max_nodes} / max_depth={max_depth}",
                    self.name
                )));
            }
            return Ok(());
        }
        let specs: Vec<TreeNodeSpec> = self
            .nodes
            .iter()
            .map(|n| TreeNodeSpec {
                parent: if n.parent < 0 { 0 } else { n.parent as u64 },
                depth: n.depth as u64,
                rank: n.rank as u64,
            })
            .collect();
        validate_draft_tree(&specs, self.spine() as u64, max_nodes, max_depth, topk)
            .map_err(|e| TreeError::Invalid(format!("{e:?}")))
    }
}

/// Builds a canonical caterpillar shape: spine `1..=spine_len` followed by leaves sorted by `(depth, rank)`.
pub fn make_shape(spine_len: usize, leaves: &[(usize, usize)], name: &str) -> TreeShape {
    let mut nodes = Vec::with_capacity(1 + spine_len + leaves.len());
    nodes.push(Node {
        parent: -1,
        depth: 0,
        rank: 0,
    });
    for j in 1..=spine_len {
        nodes.push(Node {
            parent: (j as i32) - 1,
            depth: j,
            rank: 0,
        });
    }
    let mut sorted = leaves.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    for (depth, rank) in sorted {
        if depth >= 1 && rank >= 1 {
            nodes.push(Node {
                parent: (depth as i32) - 1,
                depth,
                rank,
            });
        }
    }
    TreeShape {
        name: if name.is_empty() {
            format!("{spine_len}")
        } else {
            name.to_owned()
        },
        nodes,
        dynamic: false,
        dyn_rows: 0,
        dyn_depth: 0,
    }
}

fn parse_range(s: &str) -> Result<(usize, usize), TreeError> {
    if let Some((a, b)) = s.split_once('-') {
        let lo = a
            .parse::<usize>()
            .map_err(|_| TreeError::Parse(format!("bad range start '{s}'")))?;
        let hi = b
            .parse::<usize>()
            .map_err(|_| TreeError::Parse(format!("bad range end '{s}'")))?;
        Ok((lo, hi))
    } else {
        let v = s
            .parse::<usize>()
            .map_err(|_| TreeError::Parse(format!("bad integer '{s}'")))?;
        Ok((v, v))
    }
}

/// Parses `"D"`, `"D:j1-j2@r1-r2,..."`, or device-planned `"dN@K"`.
pub fn parse_tree_spec(spec: &str) -> Result<TreeShape, TreeError> {
    if let Some(rest) = spec.strip_prefix('d') {
        let (rows_s, depth_s) = rest
            .split_once('@')
            .ok_or_else(|| TreeError::Parse(format!("planned spec '{spec}' needs dROWS@DEPTH")))?;
        let dyn_rows = rows_s
            .parse::<usize>()
            .map_err(|_| TreeError::Parse(format!("bad rows in '{spec}'")))?;
        let dyn_depth = depth_s
            .parse::<usize>()
            .map_err(|_| TreeError::Parse(format!("bad depth in '{spec}'")))?;
        return Ok(TreeShape {
            name: spec.to_owned(),
            nodes: Vec::new(),
            dynamic: true,
            dyn_rows,
            dyn_depth,
        });
    }
    let (d_str, leaves_part) = match spec.split_once(':') {
        Some((d, rest)) => (d, Some(rest)),
        None => (spec, None),
    };
    let d = d_str
        .parse::<usize>()
        .map_err(|_| TreeError::Parse(format!("bad spine depth in '{spec}'")))?;
    let mut leaves = Vec::new();
    if let Some(rest) = leaves_part {
        for item in rest.split(',') {
            let (ds, rs) = item
                .split_once('@')
                .ok_or_else(|| TreeError::Parse(format!("tree item '{item}' needs @rank")))?;
            let (j1, j2) = parse_range(ds)?;
            let (r1, r2) = parse_range(rs)?;
            if r1 < 2 {
                return Err(TreeError::Parse(format!(
                    "tree spec '{spec}': ranks start at 2 (rank 1 is the spine)"
                )));
            }
            for j in j1..=j2 {
                for r in r1..=r2 {
                    leaves.push((j, r - 1));
                }
            }
        }
    }
    Ok(make_shape(d, &leaves, spec))
}

/// Host reference walk matching `gaudium_accept_tree`: returns accepted node row indices starting with `0`.
pub fn walk_tree(shape: &TreeShape, draft_ids: &[i32], target_tokens: &[i32]) -> Vec<usize> {
    let mut path = vec![0usize];
    let mut u = 0usize;
    loop {
        let mut found = None;
        for t in (u + 1)..shape.n_nodes() {
            if shape.nodes[t].parent == u as i32 && draft_ids[t] == target_tokens[u] {
                found = Some(t);
                break;
            }
        }
        match found {
            Some(next) => {
                u = next;
                path.push(u);
            }
            None => break,
        }
    }
    path
}

/// Online EMA acceptance policy tracker (`dtree::Stats`).
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub dmax: usize,
    pub kt: usize,
    pub alpha: f64,
    pub c: Vec<f64>,
    pub cn: Vec<usize>,
    pub q: Vec<Vec<f64>>,
    pub qn: Vec<Vec<usize>>,
    pub pool_c: f64,
    pub pool_q2: f64,
    pub steps: usize,
}

impl TreeStats {
    pub fn new(dmax: usize, kt: usize) -> Self {
        Self::with_priors(dmax, kt, 0.75, 0.65, 0.35, 0.1)
    }

    pub fn with_priors(
        dmax: usize,
        kt: usize,
        c1: f64,
        tail: f64,
        q2: f64,
        alpha: f64,
    ) -> Self {
        let mut c = vec![tail; dmax + 2];
        c[1] = c1;
        let mut q = vec![vec![0.0; kt]; dmax + 2];
        for row in &mut q {
            for (r, slot) in row.iter_mut().enumerate().skip(1) {
                *slot = q2 / (r as f64);
            }
        }
        Self {
            dmax,
            kt,
            alpha,
            c,
            cn: vec![0; dmax + 2],
            q,
            qn: vec![vec![0; kt]; dmax + 2],
            pool_c: tail,
            pool_q2: q2,
            steps: 0,
        }
    }

    pub fn observe(&mut self, shape: &TreeShape, depth: usize, last_node: usize) {
        self.steps += 1;
        let d = shape.spine();
        let is_leaf = last_node > d;
        let ds = if is_leaf { depth.saturating_sub(1) } else { depth };
        for j in 1..=d.min(ds + 1) {
            let o = if j <= ds { 1.0 } else { 0.0 };
            self.c[j] += self.alpha * (o - self.c[j]);
            self.cn[j] += 1;
            if j >= 2 {
                self.pool_c += self.alpha * (o - self.pool_c);
            }
        }
        if ds < d {
            let jb = ds + 1;
            for t in (d + 1)..shape.n_nodes() {
                if shape.nodes[t].depth == jb {
                    let r = shape.nodes[t].rank;
                    let o = if is_leaf && t == last_node { 1.0 } else { 0.0 };
                    self.q[jb][r] += self.alpha * (o - self.q[jb][r]);
                    self.qn[jb][r] += 1;
                    if r == 1 {
                        self.pool_q2 += self.alpha * (o - self.pool_q2);
                    }
                }
            }
        }
    }

    pub fn ce(&self, j: usize) -> f64 {
        if j == 1 || self.cn[j] >= 3 {
            self.c[j]
        } else {
            self.pool_c
        }
    }

    pub fn qe(&self, j: usize, r: usize) -> f64 {
        if self.qn[j][r] >= 3 {
            self.q[j][r]
        } else if r == 1 {
            self.pool_q2
        } else {
            self.q[j][r]
        }
    }

    pub fn expected_tokens(&self, shape: &TreeShape) -> f64 {
        let d = shape.spine();
        let mut a = vec![1.0f64; d + 1];
        for j in 1..=d {
            a[j] = a[j - 1] * self.ce(j.min(self.dmax + 1));
        }
        let mut e = 1.0f64;
        for &aj in &a[1..=d] {
            e += aj;
        }
        for t in (d + 1)..shape.n_nodes() {
            let j = shape.nodes[t].depth;
            if j >= 1 && j <= d {
                e += a[j - 1] * (1.0 - self.ce(j)) * self.qe(j, shape.nodes[t].rank);
            }
        }
        e
    }
}

/// Linear + online measured latency model (`dtree::Cost`).
#[derive(Debug, Clone)]
pub struct TreeCost {
    pub base_ms: f64,
    pub row_ms: f64,
    pub draft_ms: f64,
    pub tree_ms: f64,
    pub measured: BTreeMap<String, (f64, usize)>,
}

impl Default for TreeCost {
    fn default() -> Self {
        Self {
            base_ms: 24.2,
            row_ms: 0.5,
            draft_ms: 1.6,
            tree_ms: 0.6,
            measured: BTreeMap::new(),
        }
    }
}

impl TreeCost {
    pub fn model_ms(&self, rows: usize, drafted: usize, has_leaves: bool) -> f64 {
        self.base_ms
            + self.row_ms * (rows as f64)
            + self.draft_ms * (drafted as f64)
            + if has_leaves { self.tree_ms } else { 0.0 }
    }

    pub fn estimate_ms(&self, shape: &TreeShape, drafted: usize, key: &str) -> f64 {
        if let Some(&(ms, n)) = self.measured.get(key) {
            if n >= 3 {
                return ms;
            }
        }
        self.model_ms(shape.n_nodes(), drafted, shape.leaves() > 0)
    }

    pub fn observe(&mut self, key: &str, ms: f64) {
        let entry = self.measured.entry(key.to_owned()).or_insert((0.0, 0));
        if entry.1 == 0 {
            entry.0 = ms;
        } else {
            entry.0 += 0.2 * (ms - entry.0);
        }
        entry.1 += 1;
    }
}

/// Selects the candidate shape maximizing expected tokens per ms.
pub fn choose_shape(
    stats: &TreeStats,
    cands: &[TreeShape],
    cost: &TreeCost,
    filter_spine: Option<usize>,
    keys: &[&str],
) -> Option<usize> {
    let mut best = None;
    let mut best_rate = -1.0f64;
    for (i, cand) in cands.iter().enumerate() {
        if let Some(d) = filter_spine {
            if cand.spine() != d {
                continue;
            }
        }
        let rate = stats.expected_tokens(cand) / cost.estimate_ms(cand, cand.spine(), keys[i]);
        if rate > best_rate + 1e-12 {
            best_rate = rate;
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrors_tree_manager_test_cpp() {
        let c4 = parse_tree_spec("4").unwrap();
        assert_eq!((c4.n_nodes(), c4.spine(), c4.leaves()), (5, 4, 0));

        let t = parse_tree_spec("4:1-4@2").unwrap();
        assert_eq!((t.n_nodes(), t.spine(), t.leaves()), (9, 4, 4));
        for i in 5..9 {
            assert_eq!(t.nodes[i].parent, (i - 5) as i32);
            assert_eq!(t.nodes[i].depth, i - 4);
            assert_eq!(t.nodes[i].rank, 1);
        }

        let u = parse_tree_spec("5:1-2@2-3,3-5@2").unwrap();
        assert_eq!(u.n_nodes(), 6 + 4 + 3);
        for i in 7..u.n_nodes() {
            assert!(
                u.nodes[i].depth > u.nodes[i - 1].depth
                    || (u.nodes[i].depth == u.nodes[i - 1].depth
                        && u.nodes[i].rank > u.nodes[i - 1].rank)
            );
        }

        let anc = t.anc_tab();
        assert_eq!(anc[4], 0x1F);
        assert_eq!(anc[5], 1 | (1 << 5));
        assert_eq!(anc[8], 0x0F | (1 << 8));
        t.validate_caterpillar(16, 16, 4).unwrap();

        // Walk checks
        let ids = [0, 10, 11, 12, 13, 20, 21, 22, 23];
        let mut tok = [99; 9];
        tok[0] = 10;
        tok[1] = 21;
        assert_eq!(walk_tree(&t, &ids, &tok), vec![0, 1, 6]);
        tok[1] = 11;
        tok[2] = 12;
        tok[3] = 13;
        tok[4] = 5;
        assert_eq!(walk_tree(&t, &ids, &tok), vec![0, 1, 2, 3, 4]);
        tok[0] = 20;
        assert_eq!(walk_tree(&t, &ids, &tok), vec![0, 5]);

        // Adapter-style rejections (child of leaf, rank >= topk, depth > spine)
        assert!(parse_tree_spec("3:4@2").unwrap().validate_caterpillar(16, 16, 4).is_err());
        let mut g = make_shape(3, &[(1, 1)], "branch");
        g.nodes.push(Node { parent: 4, depth: 2, rank: 0 });
        assert!(g.validate_caterpillar(16, 16, 4).is_err());
        assert!(parse_tree_spec("3:1@5").unwrap().validate_caterpillar(16, 16, 4).is_err());
        assert!(parse_tree_spec("d9@4").unwrap().validate_caterpillar(16, 16, 4).is_ok());
        assert!(parse_tree_spec("d20@4").unwrap().validate_caterpillar(16, 16, 4).is_err());

        // Online policy
        let mut st = TreeStats::new(5, 4);
        let s3 = parse_tree_spec("3:1@2").unwrap();
        let e0 = st.expected_tokens(&s3);
        for _ in 0..50 {
            st.observe(&s3, 1, 4);
        }
        assert!(st.qe(1, 1) > 0.9 && st.ce(1) < 0.1);
        let c3 = parse_tree_spec("3").unwrap();
        let e1 = st.expected_tokens(&s3);
        let ec = st.expected_tokens(&c3);
        assert!(e1 > ec + 0.5, "leaf value {e1:.2} vs chain {ec:.2} (start {e0:.2})");
        let cost = TreeCost::default();
        assert_eq!(choose_shape(&st, &[c3, s3], &cost, Some(3), &["3/3", "3:1@2/3"]), Some(1));
    }
}
