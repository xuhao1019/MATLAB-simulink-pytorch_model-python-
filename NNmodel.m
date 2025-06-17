clear; clc; close all;

%% 1. Path setting (modified according to my conversion output)
model_dir = 'C:\Users\26354\Desktop\test\';
compatible_traced_path = fullfile(model_dir, 'best_transformer_model_matlab_compatible_traced.pt');
optimized_scripted_path = fullfile(model_dir, 'best_transformer_model_matlab_compatible_scripted.pt');

%% 2. Check if the model file exists
fprintf('=== Checking the converted model file ===\n');
models_to_test = {};
model_names = {};

if exist(compatible_traced_path, 'file')
    fprintf('✅ Find compatible Traced models: %s\n', compatible_traced_path);
    models_to_test{end+1} = compatible_traced_path;
    model_names{end+1} = 'Compatible Traced';
else
    fprintf('❌ Compatible Traced models not found\n');
end

if exist(optimized_scripted_path, 'file')
    fprintf('✅ Find compatible Scripted model: %s\n', optimized_scripted_path);
    models_to_test{end+1} = optimized_scripted_path;
    model_names{end+1} = 'Optimized Scripted';
else
    fprintf('❌ compatible Scripted model not found\n');
end

if isempty(models_to_test)
    fprintf('❌ No model files were found! Please check the path settings.\n');
    return;
end

%% 3. Preparing Test Data
input_dim = 15;  % Model input dimensions
test_input_raw = randn(1, input_dim, 'single');  % Single Sample Test
batch_input_raw = randn(10, input_dim, 'single');  % batch test

fprintf('\n=== Test Data Preparation ===\n');
fprintf('Single-sample input dimensions: [%s]\n', num2str(size(test_input_raw)));
fprintf('v: [%s]\n', num2str(size(batch_input_raw)));

%% 4. Test each model individually
successful_models = {};
successful_names = {};
test_results = struct();

for i = 1:length(models_to_test)
    model_path = models_to_test{i};
    model_name = model_names{i};
    
    fprintf('\n=== test %s model ===\n', model_name);
    
    try
        % load model
        net = importNetworkFromPyTorch(model_path);
        fprintf('✅ Model loaded successfully\n');
        
        % Check the network type and handle accordingly
        if isa(net, 'dlnetwork')
            fprintf('Model Type: dlnetwork\n');
            
            % Check if initialisation is required
            try
                % Attempts to predict directly, and if that fails then needs to be initialised
                test_input_dl = dlarray(test_input_raw, 'BC');
                temp_output = predict(net, test_input_dl);
                fprintf('Network initialised\n');
                initialized_net = net;
            catch
                fprintf('Need to initialise the network...\n');
                test_input_dl = dlarray(test_input_raw, 'BC');
                initialized_net = initialize(net, test_input_dl);
                fprintf('✅ Network initialisation successful\n');
            end
            
            % Single-sample prediction test
            tic;
            output_dl = predict(initialized_net, test_input_dl);
            single_time = toc;
            
            output = extractdata(output_dl);
            
        else
            fprintf('Model Type: %s\n', class(net));
            if isprop(net, 'Layers') || isfield(net, 'Layers')
                fprintf('network layer: %d\n', numel(net.Layers));
            end
            
            % Single-sample prediction test

            tic;
            output = predict(net, test_input_raw);
            single_time = toc;
            initialized_net = net;
        end
        
        fprintf('✅ Single-sample prediction success\n');
        fprintf('output dimension: [%s]\n', num2str(size(output)));
        fprintf('Projection time: %.4f秒\n', single_time);
        fprintf('output value: %.4f\n', output(1));
        
        % Batch predictive testing
        try
            if isa(initialized_net, 'dlnetwork')
                batch_input_dl = dlarray(batch_input_raw, 'BC');
                tic;
                batch_output_dl = predict(initialized_net, batch_input_dl);
                batch_time = toc;
                batch_output = extractdata(batch_output_dl);
            else
                tic;
                batch_output = predict(initialized_net, batch_input_raw);
                batch_time = toc;
            end
            
            fprintf('✅ Batch prediction success\n');
            fprintf('output dimension: [%s]\n', num2str(size(batch_output)));
            fprintf('Projection time: %.4fs (Average per sample: %.4fs)\n', ...
                batch_time, batch_time/size(batch_input_raw,1));
            
            batch_success = true;
        catch ME
            fprintf('❌ Batch prediction failure: %s\n', ME.message);
            batch_output = [];
            batch_time = NaN;
            batch_success = false;
        end
        
        % 性能基准测试
        try
            num_iterations = 50;  % 减少迭代次数以节省时间
            times = zeros(num_iterations, 1);
            
            % 准备性能测试输入
            if isa(initialized_net, 'dlnetwork')
                perf_input = dlarray(test_input_raw, 'BC');
            else
                perf_input = test_input_raw;
            end
            
            % 预热
            for j = 1:3
                predict(initialized_net, perf_input);
            end
            
            % 基准测试
            for j = 1:num_iterations
                tic;
                predict(initialized_net, perf_input);
                times(j) = toc;
            end
            
            avg_time = mean(times);
            std_time = std(times);
            min_time = min(times);
            max_time = max(times);
            
            fprintf('✅ 性能基准测试完成\n');
            fprintf('平均时间: %.4f ± %.4f秒\n', avg_time, std_time);
            fprintf('最快时间: %.4f秒, 最慢时间: %.4f秒\n', min_time, max_time);
            
            perf_success = true;
        catch ME
            fprintf('❌ 性能测试失败: %s\n', ME.message);
            avg_time = NaN;
            std_time = NaN;
            min_time = NaN;
            max_time = NaN;
            perf_success = false;
        end
        
        % 记录成功的模型
        successful_models{end+1} = initialized_net;
        successful_names{end+1} = model_name;
        
        % 保存测试结果
        test_results.(strrep(model_name, ' ', '_')) = struct(...
            'single_output', output, ...
            'single_time', single_time, ...
            'batch_output', batch_output, ...
            'batch_time', batch_time, ...
            'batch_success', batch_success, ...
            'avg_time', avg_time, ...
            'std_time', std_time, ...
            'min_time', min_time, ...
            'max_time', max_time, ...
            'perf_success', perf_success ...
        );
        
        fprintf('🎉 %s 模型测试完全成功！\n', model_name);
        
    catch ME
        fprintf('❌ %s 模型测试失败: %s\n', model_name, ME.message);
        fprintf('错误详情: %s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end
end

%% 5. 比较不同模型的输出
if length(successful_models) > 1
    fprintf('\n=== 模型输出比较 ===\n');
    
    % 比较单样本输出
    field_names = fieldnames(test_results);
    outputs = [];
    times = [];
    
    for i = 1:length(field_names)
        if isfield(test_results.(field_names{i}), 'single_output')
            output_val = test_results.(field_names{i}).single_output(1);
            time_val = test_results.(field_names{i}).single_time;
            outputs(end+1) = output_val;
            times(end+1) = time_val;
            fprintf('%s: 输出=%.4f, 时间=%.4f秒\n', ...
                strrep(field_names{i}, '_', ' '), output_val, time_val);
        end
    end
    
    if length(outputs) > 1
        output_range = max(outputs) - min(outputs);
        time_range = max(times) - min(times);
        fprintf('输出差异范围: %.6f\n', output_range);
        fprintf('时间差异范围: %.6f秒\n', time_range);
        
        if output_range < 1e-3
            fprintf('✅ 不同模型输出高度一致\n');
        elseif output_range < 1
            fprintf('⚠️  不同模型输出存在小差异\n');
        else
            fprintf('❌ 不同模型输出存在显著差异\n');
        end
    end
end

%% 6. 生成推荐和保存最佳模型
fprintf('\n=== 测试总结和推荐 ===\n');

if isempty(successful_models)
    fprintf('❌ 所有模型测试都失败了！\n');
    fprintf('建议:\n');
    fprintf('1. 检查PyTorch模型转换是否正确\n');
    fprintf('2. 确认MATLAB Deep Learning Toolbox版本\n');
    fprintf('3. 尝试使用更简单的模型结构\n');
else
    fprintf('✅ 成功测试了 %d/%d 个模型\n', length(successful_models), length(models_to_test));
    
    % 选择最佳模型（优先级：Compatible Traced > Optimized Scripted > Linear Approximation）
    best_model_idx = 1;
    best_priority = 999;
    
    for i = 1:length(successful_names)
        priority = 999;
        if contains(successful_names{i}, 'Compatible Traced')
            priority = 1;
        elseif contains(successful_names{i}, 'Optimized Scripted')
            priority = 2;
        elseif contains(successful_names{i}, 'Linear Approximation')
            priority = 3;
        end
        
        if priority < best_priority
            best_priority = priority;
            best_model_idx = i;
        end
    end
    
    best_model = successful_models{best_model_idx};
    best_name = successful_names{best_model_idx};
    
    fprintf('🏆 推荐使用: %s\n', best_name);
    
    % 保存推荐的模型
    save_path = fullfile(model_dir, 'recommended_matlab_model.mat');
    save(save_path, 'best_model', 'best_name', 'input_dim', 'test_results');
    fprintf('📁 推荐模型已保存到: %s\n', save_path);
    
    % 给出使用建议
    fprintf('\n💡 使用建议:\n');
    if contains(best_name, 'Linear Approximation')
        fprintf('⚠️  你正在使用线性近似模型，精度可能较低\n');
        fprintf('   如果精度不满足要求，建议重新训练或优化原始模型\n');
    else
        fprintf('✅ 你的Transformer模型已成功转换为MATLAB兼容格式\n');
    end
    
    fprintf('   在Simulink中使用时，确保输入数据格式为 [1, %d]\n', input_dim);
    fprintf('   模型输出格式为 [1, 1]\n');
end

fprintf('\n=== 测试完成 ===\n');