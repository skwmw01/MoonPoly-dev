import pytest
import torch
import torch.nn.functional as F
import moonpoly


def calc_diff(x,y):
    x, y = x.double(),y.double()
    denominator=(x*x+y*y).sum()
    sim=2*(x*y).sum()/denominator
    return 1-sim

class TestMoonpolyLinear:
    """Test suite for moonpoly linear layer implementation."""
    
    @pytest.fixture
    def device(self):
        """Ensure CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("batch_size,input_dim,output_dim", [
        (1, 64, 32),
        (8, 128, 64), 
        (16, 256, 128),
        (32, 512, 256),
        (16, 5120, 5120),
        (24, 5120, 25600),
        (32, 5120, 25600),
    ])
    def test_linear_correctness(self, device, dtype, batch_size, input_dim, output_dim):
        """Test moonpoly linear against torch.nn.functional.linear."""
        # Create input tensor [batch_size, input_dim]
        input_tensor = torch.randn(batch_size, input_dim, dtype=dtype, device=device)
        # Create weight tensor [output_dim, input_dim] 
        weight_tensor = torch.randn(output_dim, input_dim, dtype=dtype, device=device)
        
        # Compute reference result
        reference = F.linear(input_tensor, weight_tensor)
        
        # Compute moonpoly result
        result = moonpoly.linear(input_tensor, weight_tensor)
        
        # Verify shapes match
        assert result.shape == reference.shape, f"Shape mismatch: {result.shape} vs {reference.shape}"
        
        # Verify numerical correctness
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
    
    def test_contiguous_requirement(self, device):
        """Test that non-contiguous tensors raise appropriate errors."""
        input_tensor = torch.randn(8, 16, dtype=torch.float16, device=device).t()  # Non-contiguous
        weight_tensor = torch.randn(32, 16, dtype=torch.float16, device=device)
        
        with pytest.raises(RuntimeError, match="contiguous"):
            moonpoly.linear(input_tensor, weight_tensor)
    
    def test_device_mismatch(self):
        """Test that CPU tensors raise appropriate errors."""
        input_tensor = torch.randn(8, 16, dtype=torch.float16)  # CPU tensor
        weight_tensor = torch.randn(32, 16, dtype=torch.float16)
        
        with pytest.raises(RuntimeError, match="CUDA"):
            moonpoly.linear(input_tensor, weight_tensor)
    
    def test_dtype_mismatch(self, device):
        """Test that mismatched dtypes raise errors."""
        input_tensor = torch.randn(8, 16, dtype=torch.float16, device=device)
        weight_tensor = torch.randn(32, 16, dtype=torch.float32, device=device)  # Different dtype
        
        with pytest.raises(RuntimeError):
            moonpoly.linear(input_tensor, weight_tensor)
    
    def test_dimension_mismatch(self, device):
        """Test that incompatible dimensions raise errors."""
        input_tensor = torch.randn(8, 16, dtype=torch.float16, device=device)
        weight_tensor = torch.randn(32, 24, dtype=torch.float16, device=device)  # Wrong input dim
        
        with pytest.raises(RuntimeError):
            moonpoly.linear(input_tensor, weight_tensor)
    
    def test_output_tensor_reuse(self, device):
        """Test providing pre-allocated output tensor."""
        input_tensor = torch.randn(8, 16, dtype=torch.float16, device=device)
        weight_tensor = torch.randn(32, 16, dtype=torch.float16, device=device)
        output_tensor = torch.empty(8, 32, dtype=torch.float16, device=device)
        
        result = moonpoly.linear(input_tensor, weight_tensor, output_tensor)
        reference = F.linear(input_tensor, weight_tensor)
        
        # Should return the same tensor object
        assert result is output_tensor
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])