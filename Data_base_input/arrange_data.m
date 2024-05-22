% 定义a0值的范围和步长
a0_values = 0.1:0.1:10.0;
file_names = {'t.txt', 'x.txt', 'u.txt', 'dt.txt', 'du.txt', 'dudt.txt', 'Gm.txt'};
additional_data = [1, 0, 62.8318, 31.4159, 0, 1];
output_folder = 'CombinedData';  % 定义一个输出文件夹名称

% 确保输出文件夹存在
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 循环处理每个a0值的文件夹
for a0 = a0_values
    folder_name = sprintf('a0=%.1f', a0);
    fprintf('Processing folder: %s\n', folder_name);

    % 读取每个文件并将其内容存储为cell array
    data = cell(1, length(file_names));
    for i = 1:length(file_names)
        file_path = fullfile(folder_name, file_names{i});
        raw_data = load(file_path);
        
        % 特殊处理：如果是Gm文件，则删除第一行
        if strcmp(file_names{i}, 'Gm.txt')
            raw_data(1, :) = [];  % 删除第一行
        end
        
        data{i} = raw_data;
    end

    % 检查所有文件的行数是否一致
    num_rows = size(data{1}, 1);
    for i = 2:length(data)
        if size(data{i}, 1) ~= num_rows
            error('Row mismatch in files under folder %s', folder_name);
        end
    end

    % 组合数据
    combined_data = horzcat(data{:});

    % 添加a0值和额外的列
    a0_column = repmat(a0, num_rows, 1);
    additional_columns = repmat(additional_data, num_rows, 1);
    final_data = horzcat(combined_data, a0_column, additional_columns);

    % 保存最终数据到新文件
    output_file_name = sprintf('combined_data_a0=%.1f.txt', a0);
    output_file_path = fullfile(output_folder, output_file_name);
    save(output_file_path, 'final_data', '-ascii');
    fprintf('Data saved to %s\n', output_file_path);
end