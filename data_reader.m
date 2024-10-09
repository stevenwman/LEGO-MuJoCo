clear; clc; close all

% file = readmatrix("friction_sweep.csv");
file = readmatrix("friction_act_sweep.csv");
% file = readmatrix("act_sweep.csv");

% Example 4-column matrix (replace with your own data)
data = file;
nonzero = data~=0;
data = data(nonzero(:,1),:);

% Extract columns
colorData = data(:, 1); % First column for color
x = data(:, 2); % Second column for x
y = data(:, 3); % Third column for y

meanData = mean(colorData);
greaterIdx = colorData >= meanData;
leserIdx = colorData < meanData;

if size(data,2) == 3
    z = x*0;
else
    z = data(:, 4); % Fourth column for z
end

% Create 3D scatter plot
scatter3(x(leserIdx), y(leserIdx), z(leserIdx), 50, colorData(leserIdx), '.'); hold on
scatter3(x(greaterIdx), y(greaterIdx), z(greaterIdx), 25, colorData(greaterIdx), 'filled')

% Add labels
xlabel('param 1')
ylabel('param 2')
zlabel('param 3')
% Add colorbar
colorbar
colormap('cool')
title('Paramter Sweep')
