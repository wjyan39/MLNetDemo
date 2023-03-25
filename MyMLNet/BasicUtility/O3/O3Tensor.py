"""
Basic abstract Data Structure for SO(3) Equivariant Neural Network manipulations.
"""

import numpy as np 
import torch 
from BasicUtility.O3.O3Utility import NormContraction1d, SumsqrContraction1d, NormContraction2d, SumsqrContraction2d

from dataclasses import dataclass 
from typing import Tuple 

@dataclass
class SphericalTensor:
    """
    The SphericalTensor class tracks the SO(3) representation indices of a flattened 
    data tensor. All angular and magnetic number indices are  treated equivalent, and
    all non-degenerate feature channels are regarded as the "principal" indices, n. 

    Two indexing tensors are maintained, metadata and rep_layout.

    Args:
    data_ten :: torch.Tensor  :: The underlying data tensor.
    rep_dims :: tuple of ints :: indexes pointing to the spherical tensor representation
        dimension(s). For higher order SphericalTensor, those dimensions must be contiguous.
    metadata :: torch.LongTensor :: Specification of the number of channels for each angular
        momentum index l, in shape (n_rep_dims, n_l). 
    rep_layout :: tuple of ints :: (3, N) index tensor for (l[i], m[i], n[i]), N corresponds 
        to num_rep_dims. 
    num_channels :: tuple of ints :: number of unique channels per rep_dim.
    """
    def __init__(
        self,
        data_ten: torch.Tensor, 
        rep_dims: Tuple[int, ...],
        metadata: torch.LongTensor, 
        rep_layout: Tuple[torch.LongTensor, ...] = None, 
        num_channels: Tuple[int, ...] = None
        ):
        self.ten = data_ten 
        self.metadata = metadata 
        self.rep_dims = rep_dims 
        assert self.rep_dims[-1] < self.ten.dim() 
        self._norm_eps = 1e-4 
        if rep_layout:
            self.rep_layout = rep_layout
        else:
            if len(rep_dims) > 1:
                assert len(rep_dims) == 2
                assert rep_dims[1] - rep_dims[0] == 1
            self.rep_layout = self.generate_rep_layout() 
        if num_channels:
            self.num_channels = num_channels 
        else:
            self.num_channels = tuple(torch.sum(self.metadata, dim=1).long().tolist()) 
    
    @property
    def shape(self):
        return self.ten.shape 
    
    @property
    def device(self):
        return self.ten.device 
    
    def self_like(self, new_data_ten:torch.Tensor, this_rep_dims=None):
        """
        Generate a new SphericalTensor which has a same layout manner 
        along the rep_dims as self. 
        """
        if this_rep_dims is None:
            this_rep_dims = self.rep_dims
        return SphericalTensor(data_ten=new_data_ten, rep_dims=this_rep_dims, 
            metadata=self.metadata, rep_layout=self.rep_layout, 
            num_channels=self.num_channels)

    def mul_(self, other: "SphericalTensor"):
        # Be careful that this operation will break equivariance.
        self.ten.mul_(other.ten)
        return self.ten

    def add_(self, other: "SphericalTensor"):
        self.ten.add_(other.ten)
        return self.ten

    def __mul__(self, other: "SphericalTensor"):
        return self.self_like(self.ten * other.ten)

    def __add__(self, other: "SphericalTensor"):
        return self.self_like(self.ten + other.ten)

    def __sub__(self, other: "SphericalTensor"):
        return self.self_like(self.ten - other.ten) 

    def scalar_mul(self, other:torch.Tensor, inplace=False):
        """
        Carrying out representation-wise scalar multiplication.
        
        Args:
            other :: torch.Tensor :: along rep_dims, the shape must agree with the 
                number of unique channels, along all other dimensions shall be the 
                same as the data tensor.
            inplace :: bool :: If true, self.ten will be updated in-place.
        """
        if len(self.rep_dims) == 1:
            broadcasted_other = torch.index_select(
                # selecting the channel indexes, which is stored in self.rep_layout[0][2]
                input=other, dim=self.rep_dims[0], index=self.rep_layout[0][2,:] 
            ) 
        elif len(self.rep_dims) == 2:
            # handling meta-SphericalTensor representation, along rep_dims[0] lies sub SphericalTensors 
            broadcasted_other = torch.index_select(
                input = other.view(
                    *other.shape[:self.rep_dims[0]],             # dimensions before entering the Spherical tensor representation dimensions, e.g. batching shape
                    self.num_channels[0]*self.num_channels[1],   # flatten the true SphericalTensor data, where the operation takes place
                    *other.shape[self.rep_dims[1]+1:]            # keep the rest dimensioning
                ),
                dim = self.rep_dims[0],
                index = (self.rep_layout[0][2,:].unsqueeze(1) * self.num_channels[1] + self.rep_layout[1][2,:].unsqueeze(0)).view(-1)
            ).view_as(self.ten) # reshape as data_ten 
        else:
            raise NotImplementedError("By now, higher order SphericalTensor multipulication hasn\'t been implemented.")

        if inplace:
            self.ten.mul_(broadcasted_other) 
            return self 
        else:
            return self.self_like(self.ten * broadcasted_other)

    def dot(self, other:"SphericalTensor", dim:int):
        """
        Inner product along a representation dimension. 
        """
        dot_dim_idx = self.rep_dims.index(dim) 
        assert other.rep_dims[0] == dim 
        assert torch.all(self.metadata[dot_dim_idx].eq(other.metadata[0])) 
        out_ten = self.ten.mul(other.ten).sum(dim) 
        if len(self.rep_dims) == 1:
            return out_ten 
        elif len(self.rep_dims) == 2:
            dim_kept = 1 - dot_dim_idx 
            assert other.ten.shape[self.rep_dims[dim_kept]] == 1 
            return self.__class__(
                data_ten=out_ten,
                rep_dims=(self.rep_dims[dim_kept],),
                 metadata=self.metadata[dim_kept].unsqueeze(0),
                 rep_layout=(self.rep_layout[dim_kept],),
                 num_channels=(self.num_channels[dim_kept],)   
            )
        else:
            raise NotImplementedError 

    def rep_dot(self, other: "SphericalTensor", dim: int):
        """
        Channel-wise inner product.
        """
        dot_dim_idx = self.rep_dims.index(dim) 
        assert other.rep_dims[0] == dim 
        assert torch.all(self.metadata[dot_dim_idx].eq(other.metadata[0])) 
        mul_ten = self.ten * other.ten
        out_shape = list(mul_ten.shape) 
        out_shape[dim] = self.num_channels[dot_dim_idx] 
        out_ten = torch.zeros(out_shape, device=mul_ten.device, dtype=mul_ten.dtype).index_add_(
            dim=dim, index=self.rep_layout[dot_dim_idx][2,:], source=mul_ten 
        )
        if len(self.rep_dims) == 1:
            return out_ten 
        elif len(self.rep_dims) == 2:
            dim_kept = 1 - dot_dim_idx # the other representation dimension comes before dotting dimension 
            return self.__class__(
                data_ten=out_ten,
                rep_dims=(self.rep_dims[dim_kept],),
                metadata=self.metadata[dim_kept].unsqueeze(0),
                rep_layout=(self.rep_layout[dim_kept],),
                num_channels=(self.num_channels[dim_kept],) 
            ) 
        else:
            raise NotImplementedError 

    def rep_outer(self, other:"SphericalTensor") -> "SphericalTensor":
        """
        Outer product of two 1-d spherical tensors. a \otimes b, a 2-d SphericalTensor
        The rep_dim and metadata must be same for self and other.
        """
        assert len(self.rep_dims) == 1
        assert len(other.rep_dims) == 1 
        assert self.rep_dims[0] == other.rep_dims[0] 
        odim = self.rep_dims[0] 
        # carry out direct muliplication, A_i * B for every A[i] along dimension
        out_ten = self.ten.unsqueeze(odim+1).mul(other.ten.unsqueeze(odim)) 
        out_metadata = torch.cat([self.metadata, other.metadata],dim=0) 
        out_rep_layout = (
            self.rep_layout[0], 
            other.rep_layout[0],
        )
        return self.__class__(
            data_ten=out_ten,
            rep_dims=(odim, odim+1),
            metadata=out_metadata,
            rep_layout=out_rep_layout,
            num_channels=(self.num_channels, other.num_channels)
        ) 
    
    def fold(self, stride:int, update_self=False) -> "SphericalTensor":
        """
        Fold the chucked representation channels of a 1-d SphericalTensor to a new dimension.
        """
        assert len(self.rep_dims) == 1
        assert torch.all(torch.fmod(self.metadata[0], stride) == 0), (
            f"The number of channels for the SphericalTensor to be folded must be multiples of "
            f"stride, got ({self.metadata}, {stride}) instead"
        )
        new_ten = self.ten.unflatten(
            dim=self.rep_dims[0],
            sizes=(self.shape[self.rep_dims[0]] // stride, stride),
        )
        new_metadata = self.metadata // stride
        new_rep_layout = (self.rep_layout[0][:, ::stride],)
        new_num_channels = (self.num_channels[0] // stride,)
        if update_self:
            self.ten = new_ten
            self.metadata = new_metadata
            self.rep_layout = new_rep_layout
            self.num_channels = new_num_channels
            return self
        else:
            return self.__class__(
                new_ten,
                rep_dims=self.rep_dims,
                metadata=new_metadata,
                rep_layout=new_rep_layout,
                num_channels=new_num_channels
            )
    
    def unfold(self, update_self=False) -> "SphericalTensor":
        """
        Contract one trailing dimension into representation dimension.
        """
        assert len(self.rep_dims) == 1
        assert self.ten.dim() > self.rep_dims[0], "No trailing dimension to unfold."
        stride = self.ten.shape[self.rep_dims[0] + 1] 
        new_ten = self.ten.flatten(
            self.rep_dims[0], 
            self.rep_dims[0] + 1, 
        )
        new_metadata = self.metadata * stride 
        new_rep_layout = (self.rep_layout[0].repeat_interleave(stride, dim=1),) 
        new_num_channels = (self.num_channels[0] * stride,) 
        if update_self:
            self.ten = new_ten
            self.metadata = new_metadata
            self.rep_layout = new_rep_layout
            self.num_channels = new_num_channels
            return self
        else:
            return self.__class__(
                data_ten=new_ten,
                rep_dims=self.rep_dims,
                metadata=new_metadata,
                rep_layout=new_rep_layout,
                num_channels=new_num_channels
            )

    def transpose_repdims(self, inplace=False):
        assert (len(self.rep_dims) == 2), "transpose_repdims only supports 2d SphericalTensor"
        ten_t = torch.transpose(self.ten, *self.rep_dims).contiguous()
        dims_t = self.rep_dims
        metadata_t = self.metadata[(1, 0), :]
        rep_layout_t = self.rep_layout[::-1]
        num_channels_t = self.num_channels[::-1]
        if inplace:
            self.ten = ten_t
            self.rep_dims = dims_t
            self.metadata = metadata_t
            self.rep_layout = rep_layout_t
            self.num_channels = num_channels_t
            return self
        else:
            return self.__class__(
                ten_t,
                rep_dims=dims_t,
                metadata=metadata_t,
                rep_layout=rep_layout_t,
                num_channels=num_channels_t,
            )
    
    def invariant(self, mode="l2") -> torch.Tensor:
        """
        Returns the invariant content of a SphericalTensor
        When self.n_rep_dim==1, the l=0 channels are retained;
        when self.n_rep_dim==2, the (l1=0, l2=0) channels are also contracted.
        """
        if len(self.rep_dims) == 1:
            l0_length = self.metadata[0, 0]
            ops_dim = self.rep_dims[0] 
            data_l0 = torch.narrow(self.ten, dim=ops_dim, start=0, length=l0_length) 
            norm_shape = list(self.shape) 
            norm_shape[ops_dim] = self.num_channels[0] - l0_length
            data_rep = torch.narrow(
                self.ten, dim=ops_dim, start=l0_length, length=self.ten.shape[ops_dim] - l0_length
            ) 
            idx_ten = self.rep_layout[0][2, l0_length:].sub(l0_length) 
            if mode == "l2":
                invariant_rep = NormContraction1d.apply(data_rep, idx_ten, norm_shape, ops_dim, self._norm_eps) 
            elif mode == "uest":
                invariant_rep = NormContraction1d.apply(data_rep, idx_ten, norm_shape, ops_dim, 1.0) 
            elif mode == "sumsqr":
                invariant_rep = SumsqrContraction1d.apply(data_rep, idx_ten, norm_shape, ops_dim) 
            else:
                raise NotImplementedError 
            return torch.cat([data_l0, invariant_rep], dim=ops_dim) 
        elif len(self.rep_dims) == 2:
            idx_ten_0 = (
                self.rep_layout[0][2, :].unsqueeze(1)
                .expand(
                    self.ten.shape[self.rep_dims[0]], self.ten.shape[self.rep_dims[1]]
                )
            )
            idx_ten_1 = (
                self.rep_layout[1][2, :].unsqueeze(0)
                .expand(
                    self.ten.shape[self.rep_dims[0]], self.ten.shape[self.rep_dims[1]]
                )
            )
            idx_tens = torch.stack([idx_ten_0, idx_ten_1], dim=0) 
            norm_shape = list(self.shape) 
            norm_shape[self.rep_dims[0]] = self.num_channels[0]
            norm_shape[self.rep_dims[1]] = self.num_channels[1]
            if mode == "l2":
                invariant_2d = NormContraction2d.apply(self.ten, idx_tens, norm_shape, self.rep_dims, self._norm_eps) 
            elif mode == "uest":
                invariant_2d = NormContraction2d.apply(self.ten, idx_tens, norm_shape, self.rep_dims, 1.0) 
            elif mode == "sumsqr":
                invariant_2d = SumsqrContraction2d.apply(self.ten, idx_tens, norm_shape, self.rep_dims)
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError 
        
    def generate_rep_layout(self) -> Tuple[torch.LongTensor, ...]:
        if len(self.rep_dims) == 1:
            return (self.generate_rep_layout_1d_(self.metadata[0]).to(self.ten.device),) 
        elif len(self.rep_dims) == 2:
            rep_layout_0 = self.generate_rep_layout_1d_(self.metadata[0]).to(self.ten.device) 
            rep_layout_1 = self.generate_rep_layout_1d_(self.metadata[1]).to(self.ten.device) 
            return rep_layout_0, rep_layout_1 
        else:
            raise NotImplementedError 
    
    @staticmethod
    def generate_rep_layout_1d_(metadata1d) -> torch.LongTensor:
        # irreducible representations 2 * l + 1
        n_irreps_per_l = torch.arange(start=0, end=metadata1d.size(0)) * 2 + 1
        # generate the end offsets for each channel
        end_channel_idx = torch.cumsum(metadata1d, dim=0) 
        start_channel_idx = torch.cat([torch.LongTensor([0]), end_channel_idx[:-1]]) 
        # take metadata1d = [3, 2, 1] for instance 
        # pointing each element to a given l, angular momentum
        # [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,]
        # via [0, 1, 2] repeat times [1,3,5]*[3,2,1] = [3, 6, 5]
        dst_ls = torch.repeat_interleave(
            torch.arange(n_irreps_per_l.size(0)), n_irreps_per_l * metadata1d 
        )
        # m indexing
        # [0, 0, 0, -1, -1, 0, 0, 1, 1, -2, -1, 0, 1, 2,]
        # via [0; -1, 0, 1; -2, -1, 0, 1, 2;] repeat as [3;2,2,2;1,1,1,1,1]
        dst_ms = torch.repeat_interleave(
            torch.cat(
                [torch.arange(-l, l+1) for l in torch.arange(start=0, end=metadata1d.size(0)) ]
            ),
            torch.repeat_interleave(metadata1d, n_irreps_per_l)
        ) 
        # n indexing 
        # [0, 1, 2, 3, 4, 3, 4, 3, 4, 5, 5, 5, 5, 5] 
        ns = torch.arange(metadata1d.sum()) 
        dst_ns = torch.cat(
            [ns[start_channel_idx[l] : end_channel_idx[l]].repeat(n_irreps) 
            for l, n_irreps in enumerate(n_irreps_per_l)] 
        )
        rep_layout = torch.stack([dst_ls, dst_ms, dst_ns], dim=0).long() 
        assert rep_layout.shape[1] == torch.sum(n_irreps_per_l * metadata1d) 
        return rep_layout 
    
    def to(self, device):
        self.ten = self.ten.to(device) 
        self.rep_layout = tuple(layout.to(device) for layout in self.rep_layout)
        return self 
    

@dataclass 
class O3Tensor(SphericalTensor):
    """
    O3Tensor class tracks O(3) representation indices with additional parity (p) index.
    Inherience from SphericalTensor class.
    """
    def __init__(
        self, 
        data_ten: torch.Tensor, 
        rep_dims: Tuple[int, ...], 
        metadata: torch.LongTensor, 
        rep_layout: Tuple[torch.LongTensor, ...] = None, 
        num_channels: Tuple[int, ...] = None
    ):
        self.ten = data_ten
        self.metadata = metadata 
        self.rep_dims = rep_dims 
        self._norm_eps = 1e-4
        # each l channel has two parities, +1 and -1.
        assert self.metadata.shape[1] % 2 == 0 
        assert self.rep_dims[-1] < self.ten.dim() 
        if rep_layout:
            self.rep_layout = rep_layout 
        else:
            if len(rep_dims) > 1:
                # rep_dims for higher order SphericalTensors should be contiguous.
                assert rep_dims[1] - rep_dims[0] == 1
            self.rep_layout = self.generate_rep_layout()
        if num_channels:
            self.num_channels = num_channels 
        else:
            self.num_channels = tuple(torch.sum(self.metadata, dim=1).long().tolist())
              
        # super().__init__(data_ten, rep_dims, metadata, rep_layout, num_channels)
    
    def self_like(self, new_data_ten: torch.Tensor, this_rep_dims=None):
        if this_rep_dims is None:
            this_rep_dims = self.rep_dims
        return O3Tensor(
            new_data_ten,
            rep_dims=this_rep_dims,
            metadata=self.metadata,
            rep_layout=self.rep_layout,
            num_channels=self.num_channels
        )
    
    @staticmethod
    def generate_rep_layout_1d_(metadata1d) -> torch.LongTensor:
        num_distinct_ls = metadata1d.size(0) // 2
        n_irreps_per_l = torch.arange(start=0, end=num_distinct_ls) * 2 + 1
        n_irreps_per_lp = n_irreps_per_l.repeat_interleave(2)
        ls_metadata = torch.arange(start=0, end=num_distinct_ls).repeat_interleave(2)
        end_channel_idx = torch.cumsum(metadata1d, dim=0) 
        start_channel_idx = torch.cat([torch.LongTensor([0]), end_channel_idx[:-1]]) 
        # take metadata1d = [2, 1, 1] for instance 
        # [0, 0; 0, 0;; 1, 1, 1; 1, 1, 1;; 2, 2, 2, 2, 2; 2, 2, 2, 2, 2;;]
        dst_ls = torch.repeat_interleave(ls_metadata, n_irreps_per_lp * metadata1d)
        # m indexing
        # [0, 0; 0, 0;; -1, 0, 1; -1, 0, 1;; -2, -1, 0, 1, 2; -2, -1, 0, 1, 2;;]
        dst_ms = torch.repeat_interleave(
            torch.cat([torch.arange(-l, l+1) for l in ls_metadata]),
            torch.repeat_interleave(metadata1d, n_irreps_per_lp)
        ) 
        # n indexing 
        ns = torch.arange(metadata1d.sum()) 
        dst_ns = torch.cat(
            [ns[start_channel_idx[l] : end_channel_idx[l]].repeat(n_irreps) 
            for l, n_irreps in enumerate(n_irreps_per_lp)] 
        )
        # p indexing 
        ps = torch.LongTensor([-1, 1]).repeat(num_distinct_ls)
        dst_ps = torch.repeat_interleave(ps, n_irreps_per_lp * metadata1d)
        rep_layout = torch.stack([dst_ls, dst_ms, dst_ns, dst_ps], dim=0).long() 
        assert rep_layout.shape[1] == torch.sum(n_irreps_per_lp * metadata1d) 
        return rep_layout 
    
    @classmethod
    def from_so3(cls, so3_ten:SphericalTensor, parity=1):
        # generate O3Tensor from a SphericalTensor by extending the other parity channel.
        metadata = so3_ten.metadata 
        if parity == 1:
            # true vector 
            o3_metadata = torch.stack(
                [metadata, torch.zeros_like(metadata)], dim=2
            ).view(metadata.shape[0], -1)
            o3_layout = tuple(
                torch.cat(
                    layout, torch.ones(1, layout.shape[1], dtype=torch.long, device=layout.device), dim=0
                ) for layout in so3_ten.rep_layout
            )
        elif parity == -1:
            # pseudo vector 
            o3_metadata = torch.stack(
                [torch.zeros_like(metadata), metadata], dim=2
            ).view(metadata.shape[0], -1) 
            o3_layout = tuple(
                torch.cat(
                    layout, torch.ones(1, layout.shape[1], dtype=torch.long, device=layout.device).neg(), dim=0 
                ) for layout in so3_ten.rep_layout 
            ) 
        else:
            raise ValueError 
        
        return O3Tensor(
            data_ten=so3_ten.ten,
            rep_dims=so3_ten.rep_dims,
            metadata=o3_metadata,
            rep_layout=o3_layout,
            num_channels=so3_ten.num_channels
        )


def to_numpy(src_ten):
    # Translate to a databasing-friendly dictionary object
    if isinstance(src_ten, torch.Tensor):
        return {
            "_type": "torch.Tensor",
            "ten": src_ten.numpy(),
        }
    elif isinstance(src_ten, SphericalTensor):
        return {
            "_type": src_ten.__class__.__name__,
            "ten": src_ten.ten.numpy(),
            "rep_dims": np.asarray(src_ten.rep_dims),
            "metadata": src_ten.metadata.numpy(),
            "cated_rep_layout": torch.cat(src_ten.rep_layout, dim=1).numpy(),
            "rep_offsets": np.cumsum(
                np.asarray(list(rl.shape[1] for rl in src_ten.rep_layout))
            ),
            "num_channels": np.asarray(src_ten.num_channels),
        }
    elif src_ten is None:
        return {
            "_type": "NoneType",
        }
    else:
        raise ValueError


def from_numpy(src_dict):
    if src_dict["_type"] == "torch.Tensor":
        return torch.from_numpy(src_dict["ten"])
    elif src_dict["_type"] == "NoneType":
        return None
    else:
        target_class = getattr(BasicUtility.O3.O3Tensor, src_dict["_type"])
        rep_layout = np.split(
            src_dict["cated_rep_layout"], src_dict["rep_offsets"][:-1], axis=1
        )
        rep_layout = tuple(torch.from_numpy(rl) for rl in rep_layout)
        return target_class(
            torch.from_numpy(src_dict["ten"]),
            rep_dims=tuple(src_dict["rep_dims"]),
            metadata=torch.from_numpy(src_dict["metadata"]),
            rep_layout=rep_layout,
            num_channels=tuple(src_dict["num_channels"]),
        )



