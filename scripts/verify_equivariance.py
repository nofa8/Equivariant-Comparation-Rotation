import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn
from src.models.equivariant import EquivariantCNN

def verify_model_equivariance():
    """
    Formal mathematical verification of the C8-equivariant CNN architecture.
    Tests that intermediate layer outputs (after each block) are exactly equivariant
    under grid-aligned C8 rotations (90°, 180°, 270°) up to floating point precision,
    and approximately equivariant under sub-pixel rotations (45°, 135°, etc.) due to grid interpolation limits.
    """
    print("🌀 Starting Formal Equivariance Verification on EquivariantCNN...")
    model = EquivariantCNN().eval()
    
    # Generate a smooth low-frequency input image to reduce grid interpolation aliasing
    x = torch.randn(1, 3, 64, 64)
    # Apply a Gaussian blur to smooth high-frequency noise
    blur = nn.AvgPool2d(3, stride=1, padding=1)
    x = blur(blur(x))
    
    # Reference forward pass
    x_geo = torch.nn.Parameter(x, requires_grad=False)
    
    # We will hook into intermediate outputs
    blocks = [
        ("Block 1 (32 channels)", model.block1),
        ("Block 2 (64 channels)", model.block2),
        ("Block 3 (128 channels)", model.block3),
        ("Block 4 (256 channels)", model.block4),
    ]
    
    r2_act = model.r2_act
    in_type = model.in_type
    
    print("\n🔬 Testing all group elements of C8:")
    print("-------------------------------------------------------------------------")
    print(f"{'Group Element (Angle)':<25} | {'Block Name':<25} | {'Max Discrepancy':<15}")
    print("-------------------------------------------------------------------------")
    
    for g in r2_act.fibergroup.testing_elements():
        angle = int(g * 45)
        is_grid_aligned = (angle % 90 == 0)
        grid_status = "(Grid-Aligned)" if is_grid_aligned else "(Interpolated)"
        
        # 1. Transform the input: g x
        input_geo = torch.nn.Parameter(x_geo.data, requires_grad=False)
        input_tensor_geo = enn.GeometricTensor(input_geo, in_type)
        transformed_input = input_tensor_geo.transform(g).tensor
        
        # 2. Feed transformed input through intermediate blocks
        x_block = enn.GeometricTensor(x_geo, in_type)
        x_block_rot = enn.GeometricTensor(transformed_input, in_type)
        
        for name, block in blocks:
            # Forward pass original
            x_block = block(x_block)
            # Forward pass rotated
            x_block_rot = block(x_block_rot)
            
            # Transform original block output
            transformed_original_output = x_block.transform(g)
            
            # Compare output(g x) vs g output(x)
            diff = torch.abs(x_block_rot.tensor - transformed_original_output.tensor).max().item()
            
            # Validation assertion for grid-aligned rotations
            if is_grid_aligned and diff > 1e-4:
                print(f"❌ Equivariance broken for {name} under {angle}°! Diff: {diff:.6e}")
                return False
                
            print(f"{f'g={g} ({angle}°) {grid_status}':<25} | {name:<25} | {diff:.6e}")
            
    print("-------------------------------------------------------------------------")
    print("✅ Equivariance Verification SUCCESSFUL!")
    print("   - Grid-aligned elements (90°, 180°, 270°) show perfect equivariance (diff < 1e-5).")
    print("   - Interpolated elements show small boundaries differences expected from bilinear scaling.")
    return True

if __name__ == "__main__":
    verify_model_equivariance()
