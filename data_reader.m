clear; clc; close all

file = readmatrix("friction_sweep.csv");

% Example 4-column matrix (replace with your own data)
data = file; % 100 rows and 4 columns: [color, x, y, z]

% Extract columns
colorData = data(:, 1); % First column for color
x = data(:, 2); % Second column for x
y = data(:, 3); % Third column for y
z = data(:, 4); % Fourth column for z

% Create 3D scatter plot
scatter3(x, y, z, 50, colorData, 'filled')

% Add labels
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Z-axis')

% Add colorbar
colorbar
title('3D Scatter Plot with Color Mapped to First Column')
