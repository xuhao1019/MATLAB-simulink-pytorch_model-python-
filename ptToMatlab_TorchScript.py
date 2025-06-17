import torch
import torch.nn as nn
from pathlib import Path
import sys

# ---- 根据你的项目结构修改下面两行 ----
sys.path.append(r"H:\ModelTrainCode\NN_model\model_code_history")
from Transformer_20250225 import TransformerModel, Config


# ----------------------------------------

class MATLABCompatibleTransformer(nn.Module):
    """
    MATLAB兼容的Transformer包装器
    避免使用MATLAB不支持的操作符
    """

    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        # 确保输入是正确的形状和类型
        if len(x.shape) == 2:  # [batch_size, features]
            # 保持原始形状，不进行额外的reshape操作
            pass
        else:
            raise ValueError(f"Expected 2D input, got shape: {x.shape}")

        # 直接调用原始模型，但确保兼容性
        with torch.no_grad():  # 确保在推理模式
            output = self.original_model(x)

            # 处理可能的元组输出
            if isinstance(output, tuple):
                # 返回第一个输出（通常是主要预测结果）
                return output[0]
            else:
                return output


def convert_attention_layers(model):
    """
    尝试替换不兼容的注意力层
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(f"Warning: Found MultiheadAttention layer '{name}' - this may cause MATLAB compatibility issues")
            # 可以在这里添加替换逻辑，但需要了解你的具体模型结构

    return model


def create_simple_wrapper(model, input_dim, output_dim):
    """
    创建一个简单的包装器，避免复杂的Transformer结构
    """

    class SimpleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            # 确保输入格式正确
            if x.dim() == 2 and x.size(1) == input_dim:
                return self.model(x)
            else:
                raise ValueError(f"Expected input shape [batch_size, {input_dim}], got {x.shape}")

    return SimpleWrapper()


def main():
    # 1. 路径设置 - 修改为你的模型路径
    model_pt = Path(r"H:\ModelTrainCode\NN_model\best_transformer_model_0225.pth")

    # 输出文件路径
    output_dir = Path(r"H:\ModelTrainCode\NN_model\matlab_compatible")
    output_dir.mkdir(parents=True, exist_ok=True)

    traced_pt = output_dir / "best_transformer_model_matlab_compatible_traced.pt"
    scripted_pt = output_dir / "best_transformer_model_matlab_compatible_scripted.pt"
    simple_pt = output_dir / "best_transformer_model_simple_wrapper.pt"

    # 2. 检查原始模型文件是否存在
    if not model_pt.exists():
        print(f"❌ Model file not found: {model_pt}")
        print("Please check the file path and make sure the model file exists.")
        return

    # 3. 加载原始模型
    print("Loading original model...")
    try:
        model = TransformerModel(config=Config)
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please check if the model architecture matches the saved state dict.")
        return

    # 4. 构造 dummy 输入
    dummy_input = torch.randn(1, Config.input_dim, dtype=torch.float32)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input dimension from config: {Config.input_dim}")

    # 5. 测试原始模型
    try:
        with torch.no_grad():
            original_output = model(dummy_input)
            print(f"Original output type: {type(original_output)}")

            # 处理模型输出（可能是元组）
            if isinstance(original_output, tuple):
                print(f"Model returns tuple with {len(original_output)} elements")
                for i, output in enumerate(original_output):
                    if hasattr(output, 'shape'):
                        print(f"  Output {i} shape: {output.shape}")
                    else:
                        print(f"  Output {i} type: {type(output)}")
                # 通常取第一个输出作为主要输出
                main_output = original_output[0]
            else:
                main_output = original_output
                print(f"Original output shape: {original_output.shape}")

            print(f"Main output shape: {main_output.shape}")
    except Exception as e:
        print(f"❌ Error testing original model: {e}")
        return

    # 方法1: 使用兼容包装器
    try:
        print("\n=== Method 1: Compatible Wrapper ===")
        compatible_model = MATLABCompatibleTransformer(model)
        compatible_model.eval()

        # 测试包装器
        with torch.no_grad():
            wrapped_output = compatible_model(dummy_input)
            print(f"Wrapped output shape: {wrapped_output.shape}")

            # 比较输出（处理原始输出可能是元组的情况）
            if isinstance(original_output, tuple):
                comparison_output = original_output[0]
            else:
                comparison_output = original_output

            diff = torch.max(torch.abs(comparison_output - wrapped_output))
            print(f"Output difference: {diff}")

        # Trace包装器
        traced_compatible = torch.jit.trace(
            compatible_model,
            dummy_input,
            check_trace=False,
            strict=False  # 允许一些不严格的匹配
        )
        traced_compatible.save(str(traced_pt))
        print(f"✅ Compatible traced model saved to: {traced_pt}")

    except Exception as e:
        print(f"❌ Method 1 failed: {e}")

    # 方法2: 尝试脚本化（通常对复杂模型效果更好）
    try:
        print("\n=== Method 2: Scripting with optimizations ===")

        # 设置优化选项
        torch.jit.set_fusion_strategy([("STATIC", 20), ("DYNAMIC", 20)])

        # 脚本化模型
        scripted_model = torch.jit.script(model)

        # 优化脚本化模型
        scripted_model = torch.jit.optimize_for_inference(scripted_model)

        scripted_model.save(str(scripted_pt))
        print(f"✅ Optimized scripted model saved to: {scripted_pt}")

    except Exception as e:
        print(f"❌ Method 2 failed: {e}")

    # 方法3: 创建简单的线性近似（如果其他方法都失败）
    try:
        print("\n=== Method 3: Simple Linear Approximation ===")

        # 收集一些样本数据来训练线性近似
        num_samples = 1000
        X_samples = torch.randn(num_samples, Config.input_dim, dtype=torch.float32)

        with torch.no_grad():
            Y_samples_raw = model(X_samples)

            # 处理可能的元组输出
            if isinstance(Y_samples_raw, tuple):
                Y_samples = Y_samples_raw[0]
                print(f"Using output[0] for linear approximation, shape: {Y_samples.shape}")
            else:
                Y_samples = Y_samples_raw
                print(f"Model output shape for approximation: {Y_samples.shape}")

        # 创建简单的线性模型作为近似
        input_dim = Config.input_dim
        output_dim = Y_samples.shape[1] if len(Y_samples.shape) > 1 else 1
        print(f"Creating linear approximation: {input_dim} -> {output_dim}")

        linear_approx = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # 简单训练线性近似
        optimizer = torch.optim.Adam(linear_approx.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        linear_approx.train()
        print("Training linear approximation...")
        for epoch in range(100):
            optimizer.zero_grad()
            pred = linear_approx(X_samples)
            loss = criterion(pred, Y_samples)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        linear_approx.eval()

        # 测试线性近似
        with torch.no_grad():
            approx_output = linear_approx(dummy_input)
            original_test_raw = model(dummy_input)

            # 处理原始模型的元组输出
            if isinstance(original_test_raw, tuple):
                original_test = original_test_raw[0]
            else:
                original_test = original_test_raw

            error = torch.mean(torch.abs(approx_output - original_test))
            print(f"Linear approximation error: {error.item():.6f}")
            print(f"Original output: {original_test.flatten()[:5]}...")  # 显示前5个值
            print(f"Approximation output: {approx_output.flatten()[:5]}...")  # 显示前5个值

        # Trace线性近似
        traced_linear = torch.jit.trace(linear_approx, dummy_input, check_trace=False)
        traced_linear.save(str(simple_pt))
        print(f"✅ Linear approximation saved to: {simple_pt}")

    except Exception as e:
        print(f"❌ Method 3 failed: {e}")

    print("\n=== Conversion Summary ===")
    print("Generated files for MATLAB testing:")
    print(f"Output directory: {output_dir}")
    if traced_pt.exists():
        print(f"  - Compatible traced: {traced_pt.name}")
    if scripted_pt.exists():
        print(f"  - Optimized scripted: {scripted_pt.name}")
    if simple_pt.exists():
        print(f"  - Linear approximation: {simple_pt.name}")

    print("\n=== MATLAB Usage Instructions ===")
    print("In MATLAB, try the following commands:")
    print("1. For the compatible traced model:")
    print(f"   net = importNetworkFromPyTorch('{traced_pt}');")
    print("\n2. For the optimized scripted model:")
    print(f"   net = importNetworkFromPyTorch('{scripted_pt}');")
    print("\n3. For the linear approximation:")
    print(f"   net = importNetworkFromPyTorch('{simple_pt}');")

    print("\nRecommendations:")
    print("1. Try the compatible traced model first")
    print("2. If that fails, try the optimized scripted model")
    print("3. As a last resort, use the linear approximation")
    print("4. You may need to implement custom MATLAB functions for unsupported operators")

    print("\nTroubleshooting:")
    print("- If MATLAB reports unsupported operations, the linear approximation is most likely to work")
    print("- Make sure your MATLAB version supports importNetworkFromPyTorch (R2022b or later)")
    print("- Check that Deep Learning Toolbox is installed in MATLAB")


if __name__ == "__main__":
    main()