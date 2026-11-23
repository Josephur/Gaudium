//! GGUF metadata value types and ggml tensor types: format facts of the GGUF specification
//! (ids, element sizes, block geometry), kept as tables and pinned by tests.

/// A metadata value type (GGUF value type ids 0..=12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub fn from_id(id: u32) -> Option<Self> {
        use ValueType::*;
        Some(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => Str,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }
    /// Bytes of one value of a fixed-size type (`None` for strings and arrays).
    pub fn fixed_size(self) -> Option<u64> {
        use ValueType::*;
        Some(match self {
            U8 | I8 | Bool => 1,
            U16 | I16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
            Str | Array => return None,
        })
    }
    pub fn is_float(self) -> bool {
        matches!(self, ValueType::F32 | ValueType::F64)
    }
    pub fn name(self) -> &'static str {
        use ValueType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            Bool => "bool",
            Str => "str",
            Array => "arr",
            U64 => "u64",
            I64 => "i64",
            F64 => "f64",
        }
    }
}

/// Block geometry of a ggml tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeInfo {
    pub id: u32,
    pub name: &'static str,
    /// Weights per block (1 for plain number types).
    pub block_weights: u64,
    /// Bytes per block.
    pub block_bytes: u64,
    /// The host decoder can turn this type into numbers.
    pub decodable: bool,
}

const fn ti(id: u32, name: &'static str, bw: u64, bb: u64, decodable: bool) -> TypeInfo {
    TypeInfo {
        id,
        name,
        block_weights: bw,
        block_bytes: bb,
        decodable,
    }
}

/// Every ggml type id a GGUF file may carry (removed ids 4, 5, 31..=33, 36..=38 are absent).
/// The grid-codebook types (IQ1/IQ2/IQ3, TQ, Q1_0, Q2_0) are known to the container so that
/// a file holding them opens and can be inspected, but they are not decoded yet.
pub const TYPES: &[TypeInfo] = &[
    ti(0, "F32", 1, 4, true),
    ti(1, "F16", 1, 2, true),
    ti(2, "Q4_0", 32, 18, true),
    ti(3, "Q4_1", 32, 20, true),
    ti(6, "Q5_0", 32, 22, true),
    ti(7, "Q5_1", 32, 24, true),
    ti(8, "Q8_0", 32, 34, true),
    ti(9, "Q8_1", 32, 36, true),
    ti(10, "Q2_K", 256, 84, true),
    ti(11, "Q3_K", 256, 110, true),
    ti(12, "Q4_K", 256, 144, true),
    ti(13, "Q5_K", 256, 176, true),
    ti(14, "Q6_K", 256, 210, true),
    ti(15, "Q8_K", 256, 292, true),
    ti(16, "IQ2_XXS", 256, 66, false),
    ti(17, "IQ2_XS", 256, 74, false),
    ti(18, "IQ3_XXS", 256, 98, false),
    ti(19, "IQ1_S", 256, 50, false),
    ti(20, "IQ4_NL", 32, 18, true),
    ti(21, "IQ3_S", 256, 110, false),
    ti(22, "IQ2_S", 256, 82, false),
    ti(23, "IQ4_XS", 256, 136, true),
    ti(24, "I8", 1, 1, true),
    ti(25, "I16", 1, 2, true),
    ti(26, "I32", 1, 4, true),
    ti(27, "I64", 1, 8, true),
    ti(28, "F64", 1, 8, true),
    ti(29, "IQ1_M", 256, 56, false),
    ti(30, "BF16", 1, 2, true),
    ti(34, "TQ1_0", 256, 54, false),
    ti(35, "TQ2_0", 256, 66, false),
    ti(39, "MXFP4", 32, 17, true),
    ti(40, "NVFP4", 64, 36, true),
    ti(41, "Q1_0", 128, 18, false),
    ti(42, "Q2_0", 64, 18, false),
    ti(43, "F8_E4M3", 1, 1, true),
];

/// A ggml tensor type known to this reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GgmlType(u32);

impl GgmlType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K: Self = Self(11);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const Q6_K: Self = Self(14);
    pub const Q8_K: Self = Self(15);
    pub const IQ4_NL: Self = Self(20);
    pub const IQ4_XS: Self = Self(23);
    pub const I8: Self = Self(24);
    pub const I16: Self = Self(25);
    pub const I32: Self = Self(26);
    pub const I64: Self = Self(27);
    pub const F64: Self = Self(28);
    pub const BF16: Self = Self(30);
    pub const MXFP4: Self = Self(39);
    pub const NVFP4: Self = Self(40);
    pub const F8_E4M3: Self = Self(43);

    pub fn from_id(id: u32) -> Option<Self> {
        TYPES.iter().any(|t| t.id == id).then_some(Self(id))
    }
    pub fn from_name(name: &str) -> Option<Self> {
        TYPES.iter().find(|t| t.name == name).map(|t| Self(t.id))
    }
    pub fn id(self) -> u32 {
        self.0
    }
    pub fn info(self) -> &'static TypeInfo {
        // constructed only from ids in TYPES
        TYPES.iter().find(|t| t.id == self.0).unwrap_or(&TYPES[0])
    }
    pub fn name(self) -> &'static str {
        self.info().name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_table_matches_the_format() {
        // (id, weights per block, bytes per block): the GGUF specification's block sizes
        let want: &[(u32, u64, u64)] = &[
            (0, 1, 4),
            (1, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (20, 32, 18),
            (23, 256, 136),
            (30, 1, 2),
            (39, 32, 17),
            (40, 64, 36),
            (43, 1, 1),
        ];
        for &(id, bw, bb) in want {
            let t = GgmlType::from_id(id).unwrap().info();
            assert_eq!((t.block_weights, t.block_bytes), (bw, bb), "{}", t.name);
        }
        for id in [4, 5, 31, 32, 33, 36, 37, 38, 44] {
            assert!(GgmlType::from_id(id).is_none(), "{id}");
        }
        assert_eq!(GgmlType::from_name("NVFP4"), Some(GgmlType::NVFP4));
    }

    #[test]
    fn value_type_sizes() {
        let sizes: Vec<Option<u64>> = (0..13)
            .map(|i| ValueType::from_id(i).unwrap().fixed_size())
            .collect();
        assert_eq!(
            sizes,
            vec![
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(4),
                Some(4),
                Some(4),
                Some(1),
                None,
                None,
                Some(8),
                Some(8),
                Some(8)
            ]
        );
        assert!(ValueType::from_id(13).is_none());
    }
}
