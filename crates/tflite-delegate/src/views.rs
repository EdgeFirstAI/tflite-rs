// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Strided locations of tensors in the delegate's buffers, and the view operators on them.

/// Which memory a location is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Buf {
    /// Partition input `k` (including model constants).
    Input(usize),
    /// Partition output `k`.
    Output(usize),
    /// Delegate scratch buffer `k`.
    Scratch(usize),
}

/// A strided view of a buffer: element `idx` is at `off + Σ idx[a]·strides[a]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Loc {
    pub(crate) buf: Buf,
    pub(crate) off: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
}

/// Row-major strides of `shape`.
pub(crate) fn dense_strides(shape: &[usize]) -> Vec<isize> {
    let mut s = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for (st, &d) in s.iter_mut().zip(shape).rev() {
        *st = acc;
        acc *= d.cast_signed();
    }
    s
}

impl Loc {
    /// The whole of `buf` as a dense tensor of `shape`.
    pub(crate) fn dense(buf: Buf, shape: &[usize]) -> Self {
        Self {
            buf,
            off: 0,
            shape: shape.to_vec(),
            strides: dense_strides(shape),
        }
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// One past the largest element offset the view addresses (0 when empty), or `None` if
    /// that overflows.
    pub(crate) fn extent(&self) -> Option<usize> {
        if self.shape.contains(&0) {
            return Some(0);
        }
        self.shape
            .iter()
            .zip(&self.strides)
            .try_fold(self.off, |acc, (&d, &s)| {
                (d - 1)
                    .checked_mul(s.unsigned_abs())
                    .and_then(|span| acc.checked_add(span))
            })?
            .checked_add(1)
    }

    /// Whether the view is dense row-major from its offset.
    #[cfg(test)]
    pub(crate) fn is_dense(&self) -> bool {
        self.shape
            .iter()
            .zip(&self.strides)
            .zip(dense_strides(&self.shape))
            .all(|((&d, &s), e)| d == 1 || s == e)
    }

    /// `TRANSPOSE`: output axis `a` is input axis `perm[a]`.
    pub(crate) fn transpose(&self, perm: &[usize]) -> Option<Self> {
        if perm.len() != self.shape.len() {
            return None;
        }
        Some(Self {
            buf: self.buf,
            off: self.off,
            shape: perm
                .iter()
                .map(|&p| self.shape.get(p).copied())
                .collect::<Option<_>>()?,
            strides: perm
                .iter()
                .map(|&p| self.strides.get(p).copied())
                .collect::<Option<_>>()?,
        })
    }

    /// The last two axes swapped (a matrix transpose).
    pub(crate) fn swap_last2(&self) -> Option<Self> {
        let r = self.shape.len();
        if r < 2 {
            return None;
        }
        let mut perm: Vec<usize> = (0..r).collect();
        perm.swap(r - 2, r - 1);
        self.transpose(&perm)
    }

    /// `SLICE` of `size` elements from `begin` on each axis.
    pub(crate) fn slice(&self, begin: &[usize], size: &[usize]) -> Option<Self> {
        if begin.len() != self.shape.len() || size.len() != self.shape.len() {
            return None;
        }
        let mut off = self.off;
        for ((&b, &n), (&d, &s)) in begin
            .iter()
            .zip(size)
            .zip(self.shape.iter().zip(&self.strides))
        {
            if b + n > d {
                return None;
            }
            off += b * s.unsigned_abs();
        }
        Some(Self {
            buf: self.buf,
            off,
            shape: size.to_vec(),
            strides: self.strides.clone(),
        })
    }

    /// `n` elements from `begin` along `axis`.
    pub(crate) fn narrow(&self, axis: usize, begin: usize, n: usize) -> Option<Self> {
        let mut b = vec![0; self.shape.len()];
        let mut size = self.shape.clone();
        *b.get_mut(axis)? = begin;
        *size.get_mut(axis)? = n;
        self.slice(&b, &size)
    }

    /// The same elements as `shape`, without copying, when the strides allow it.
    ///
    /// Size-1 axes are dropped, then runs of input axes are matched to runs of output axes
    /// with the same element count; each input run must be contiguous within itself.
    pub(crate) fn reshape(&self, shape: &[usize]) -> Option<Self> {
        let n: usize = shape.iter().product();
        if n != self.numel() {
            return None;
        }
        if n == 0 {
            return Some(Self {
                buf: self.buf,
                off: self.off,
                shape: shape.to_vec(),
                strides: dense_strides(shape),
            });
        }
        let old: Vec<(usize, isize)> = self
            .shape
            .iter()
            .zip(&self.strides)
            .filter(|(&d, _)| d != 1)
            .map(|(&d, &s)| (d, s))
            .collect();
        let new_axes: Vec<usize> = (0..shape.len()).filter(|&a| shape[a] != 1).collect();
        let mut strides = vec![0isize; shape.len()];
        let (mut i, mut j) = (0usize, 0usize);
        while i < old.len() && j < new_axes.len() {
            // Grow both runs until their element counts match.
            let (i0, j0) = (i, j);
            let (mut pi, mut pj) = (old[i].0, shape[new_axes[j]]);
            while pi != pj {
                if pi < pj {
                    i += 1;
                    pi *= old.get(i)?.0;
                } else {
                    j += 1;
                    pj *= shape[*new_axes.get(j)?];
                }
            }
            // The input run must be contiguous within itself.
            for k in i0..i {
                if old[k].1 != old[k + 1].1 * old[k + 1].0.cast_signed() {
                    return None;
                }
            }
            // Output run strides, innermost first.
            let mut st = old[i].1;
            for k in (j0..=j).rev() {
                let a = new_axes[k];
                strides[a] = st;
                st *= shape[a].cast_signed();
            }
            i += 1;
            j += 1;
        }
        if i != old.len() || j != new_axes.len() {
            return None;
        }
        Some(Self {
            buf: self.buf,
            off: self.off,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Broadcast to `shape`, right-aligned: size-1 (and missing leading) axes get stride 0.
    pub(crate) fn broadcast_to(&self, shape: &[usize]) -> Option<Self> {
        if self.shape.len() > shape.len() {
            // Leading size-1 axes may be dropped.
            let extra = self.shape.len() - shape.len();
            if self.shape[..extra].iter().any(|&d| d != 1) {
                return None;
            }
            return Self {
                buf: self.buf,
                off: self.off,
                shape: self.shape[extra..].to_vec(),
                strides: self.strides[extra..].to_vec(),
            }
            .broadcast_to(shape);
        }
        let lead = shape.len() - self.shape.len();
        let mut strides = vec![0isize; shape.len()];
        for (a, &d) in shape.iter().enumerate() {
            if a < lead {
                continue;
            }
            let (sd, ss) = (self.shape[a - lead], self.strides[a - lead]);
            if sd == d {
                strides[a] = ss;
            } else if sd != 1 {
                return None;
            }
        }
        Some(Self {
            buf: self.buf,
            off: self.off,
            shape: shape.to_vec(),
            strides,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elems(l: &Loc) -> Vec<usize> {
        let mut out = Vec::new();
        let mut idx = vec![0usize; l.shape.len()];
        if l.numel() == 0 {
            return out;
        }
        loop {
            out.push(
                l.off
                    + idx
                        .iter()
                        .zip(&l.strides)
                        .map(|(&i, &s)| i * s.cast_unsigned())
                        .sum::<usize>(),
            );
            let mut a = l.shape.len();
            loop {
                if a == 0 {
                    return out;
                }
                a -= 1;
                idx[a] += 1;
                if idx[a] < l.shape[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
    }

    #[test]
    fn reshape_of_dense_is_dense() {
        let l = Loc::dense(Buf::Scratch(0), &[2, 3, 4]);
        let r = l.reshape(&[6, 4]).unwrap();
        assert!(r.is_dense());
        assert_eq!(elems(&r), elems(&l));
        assert_eq!(elems(&l.reshape(&[1, 24, 1]).unwrap()), elems(&l));
    }

    #[test]
    fn reshape_splits_a_transposed_axis() {
        // [400, 256] transposed to [256, 400], then split 256 into 2 x 128 (yolo26).
        let x = Loc::dense(Buf::Input(0), &[400, 256]);
        let t = x.transpose(&[1, 0]).unwrap();
        let r = t.reshape(&[1, 2, 128, 400]).unwrap();
        assert_eq!(r.strides, vec![0, 128, 1, 256]);
        assert_eq!(elems(&r), elems(&t));
        // Merging the transposed axes back is impossible without a copy.
        assert!(t.reshape(&[102_400]).is_none());
    }

    #[test]
    fn transpose_slice_and_broadcast() {
        let x = Loc::dense(Buf::Input(0), &[4, 6]);
        let s = x.slice(&[1, 2], &[2, 3]).unwrap();
        assert_eq!(elems(&s), vec![8, 9, 10, 14, 15, 16]);
        let b = Loc::dense(Buf::Input(1), &[1, 6])
            .broadcast_to(&[4, 6])
            .unwrap();
        assert_eq!(b.strides, vec![0, 1]);
        let lead = Loc::dense(Buf::Input(1), &[1, 1, 4, 6])
            .broadcast_to(&[4, 6])
            .unwrap();
        assert_eq!(lead.strides, vec![6, 1]);
        assert!(Loc::dense(Buf::Input(1), &[2, 6])
            .broadcast_to(&[4, 6])
            .is_none());
        assert_eq!(x.narrow(1, 4, 2).unwrap().off, 4);
        assert_eq!(x.swap_last2().unwrap().strides, vec![1, 6]);
    }
}
