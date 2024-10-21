clear; clc; close all

% file = readmatrix("friction_sweep.csv");
% file = readmatrix("friction_act_sweep.csv");
file = readmatrix("act_sweep.csv");
% file = readmatrix("contact_freq_sweep.csv");

% Example 4-column matrix (replace with your own data)
data = file;
nonzero = data~=0;
data = data(nonzero(:,1),:);

exp_dist = 2.3;

error = abs(data(:, 1) - exp_dist);

% Extract columns
colorData = data(:, 1); % First column for color
% colorData = error;
x = data(:, 2); % Second column for x
y = data(:, 3); % Third column for y

meanData = mean(colorData);
% greaterIdx = colorData >= meanData;
% leserIdx = colorData < meanData;

% LowErrIdx = colorData <= meanData;
% HighErrIdx = colorData > meanData;

if size(data,2) == 3
    z = x*0;
else
    z = data(:, 4); % Fourth column for z
end

% Create 3D scatter plot
% scatter3(x(leserIdx), y(leserIdx), z(leserIdx), 50, colorData(leserIdx), '.'); hold on
% scatter3(x(greaterIdx), y(greaterIdx), z(greaterIdx), 25, colorData(greaterIdx), 'filled')

% scatter3(x(HighErrIdx), y(HighErrIdx), z(HighErrIdx), 50, colorData(HighErrIdx), '.'); hold on
% scatter3(x(LowErrIdx), y(LowErrIdx), z(LowErrIdx), 25, colorData(LowErrIdx), 'filled')

scatter3(x,y,z,25,colorData, 'filled')

% Add labels
xlabel('param 1')
ylabel('param 2')
zlabel('param 3')
% Add colorbar
colorbar
colormap('cool')
title('Paramter Sweep')
