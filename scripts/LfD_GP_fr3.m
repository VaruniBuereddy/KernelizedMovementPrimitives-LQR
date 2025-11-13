% MATLAB script equivalent to the provided Python script

% Clear workspace and close all figures
clear; clc; close all;

% Check if the file exists and delete it
%if exist('STT_arm_LfD.csv', 'file') == 2
%    delete('STT_arm_LfD.csv');
%end

% Read the data from the CSV file
A = readmatrix('joint_positions_demo_tst.csv');

% Extract columns from the data
global theta0
theta0 = A(1,:);
td = A(:, 1);
j1 = A(:, 2)-theta0(2);
j2 = A(:, 3)-theta0(3);
j3 = A(:, 4)-theta0(4);
j4 = A(:, 5)-theta0(5);
j5 = A(:, 6)-theta0(6);
j6 = A(:, 7)-theta0(7);
j7 = A(:, 8)-theta0(8);
x = A(:, 9);
y = A(:, 10);
z = A(:, 11);

% Generate noisy data points
tg = td;
j1g = j1;
j2g = j2;
j3g = j3;
j4g = j4;
j5g = j5;
j6g = j6;
j7g = j7;
% xg = x;
% yg = y;
% zg = z;

% Add flipped data with noise
d = 0;
tg = [tg; flip(td)];
j1g = [j1g; flip(j1) + 2 * d * rand(size(td)) - d];
j2g = [j2g; flip(j2) + 2 * d * rand(size(td)) - d];
j3g = [j3g; flip(j3) + 2 * d * rand(size(td)) - d];
j4g = [j4g; flip(j4) + 2 * d * rand(size(td)) - d];
j5g = [j5g; flip(j5) + 2 * d * rand(size(td)) - d];
j6g = [j6g; flip(j6) + 2 * d * rand(size(td)) - d];
j7g = [j7g; flip(j7) + 2 * d * rand(size(td)) - d];
% xg = [xg; flip(x) + 2 * d * rand(size(td)) - d];
% yg = [yg; flip(y) + 2 * d * rand(size(td)) - d];
% zg = [zg; flip(z) + 2 * d * rand(size(td)) - d];

% Gaussian Process Regression
sigma0 = 0.1;

% Define custom kernel function with KernelParameters
kernel_test = @(X1, X2, theta) theta(2)^2 * exp(-0.5 * pdist2(X1, X2).^2 / theta(1)^2);

% Initial kernel parameters: [length_scale, signal_variance]
initial_theta = [3.5; 6.2]; % Example values

% Fit GPR models for each joint
gprMdlj1 = fitrgp(tg, j1g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);
gprMdlj2 = fitrgp(tg, j2g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);
gprMdlj3 = fitrgp(tg, j3g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);
gprMdlj4 = fitrgp(tg, j4g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);
gprMdlj5 = fitrgp(tg, j5g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);
gprMdlj6 = fitrgp(tg, j6g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);
gprMdlj7 = fitrgp(tg, j7g, 'KernelFunction', kernel_test, 'KernelParameters', initial_theta, 'Sigma', sigma0);

% Predict using the GPR models
[j1pred, ~, j1ci] = predict(gprMdlj1, tg);
[j2pred, ~, j2ci] = predict(gprMdlj2, tg);
[j3pred, ~, j3ci] = predict(gprMdlj3, tg);
[j4pred, ~, j4ci] = predict(gprMdlj4, tg);
[j5pred, ~, j5ci] = predict(gprMdlj5, tg);
[j6pred, ~, j6ci] = predict(gprMdlj6, tg);
[j7pred, ~, j7ci] = predict(gprMdlj7, tg);

% Shift of origin
% j1pred = j1pred + theta0(2);
% j2pred = j2pred + theta0(3);
% j3pred = j3pred + theta0(4);
% j4pred = j4pred + theta0(5);
% j5pred = j5pred + theta0(6);
% j6pred = j6pred + theta0(7);
% j7pred = j7pred + theta0(8);
% 
% j1g = j1g + theta0(2);
% j2g = j2g + theta0(3);
% j3g = j3g + theta0(4);
% j4g = j4g + theta0(5);
% j5g = j5g + theta0(6);
% j6g = j6g + theta0(7);
% j7g = j7g + theta0(8);

% Plot results
figure;
subplot(7, 1, 1);
plot(tg, j1g, 'k.', tg, j1pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j1'); legend('Data', 'GP Predictions');

subplot(7, 1, 2);
plot(tg, j2g, 'k.', tg, j2pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j2'); legend('Data', 'GP Predictions');

subplot(7, 1, 3);
plot(tg, j3g, 'k.', tg, j3pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j3'); legend('Data', 'GP Predictions');

subplot(7, 1, 4);
plot(tg, j4g, 'k.', tg, j4pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j4'); legend('Data', 'GP Predictions');

subplot(7, 1, 5);
plot(tg, j5g, 'k.', tg, j5pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j5'); legend('Data', 'GP Predictions');

subplot(7, 1, 6);
plot(tg, j6g, 'k.', tg, j6pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j6'); legend('Data', 'GP Predictions');

subplot(7, 1, 7);
plot(tg, j7g, 'k.', tg, j7pred, 'g-', 'LineWidth', 1.5);
xlabel('t'); ylabel('j7'); legend('Data', 'GP Predictions');

% Spatiotemporal Tubes (STT) for each joint
td_new = linspace(0, td(end), 1000)';
STT = zeros(length(td_new), 14); % 7 joints * 2 (lower and upper bounds) 

%%
figure
plot3(x,y,z, 'b--', 'LineWidth',1.5)
grid on
axis square
xlabel('x')
ylabel('y')
zlabel('z')
% view([30 30])
%%
% Function to compute STT for each joint
function STT = computeSTT(gprMdl, tg, td_new, joint_data, STT, col_idx, add_thk)
    global theta0
    sigmaL = gprMdl.KernelInformation.KernelParameters(1);
    sigmaF = gprMdl.KernelInformation.KernelParameters(2);
    N = length(tg);
    
    % Covariance matrix K
    K = sigmaF^2 * exp(-0.5 * pdist2(tg, tg).^2 / sigmaL^2);
    I = eye(N);
    rhoF = 0.01;
    
    % Compute mu_t and rho_t
    mu_t = zeros(length(td_new), 1);
    rho_t = zeros(length(td_new), 1);
    
    for i = 1:length(td_new)
        kb_t = sigmaF^2 * exp(-0.5 * (tg - td_new(i)).^2 / sigmaL^2);
        ktt = sigmaF^2;
        mu_t(i) = kb_t' * inv(K + rhoF^2 * I) * joint_data;
        rho_t(i) = sqrt(ktt - kb_t' * inv(K + rhoF^2 * I) * kb_t) + add_thk;
    end
    
    % Store results in STT matrix
    STT(:, col_idx) = mu_t - rho_t  + theta0((col_idx+3)/2);
    STT(:, col_idx+1) = mu_t + rho_t  + theta0((col_idx+3)/2);
end

% Compute STT for each joint
STT = computeSTT(gprMdlj1, tg, td_new, j1g, STT, 1, 0.05);
STT = computeSTT(gprMdlj2, tg, td_new, j2g, STT, 3, 0.05);
STT = computeSTT(gprMdlj3, tg, td_new, j3g, STT, 5, 0.05);
STT = computeSTT(gprMdlj4, tg, td_new, j4g, STT, 7, 0.05);
STT = computeSTT(gprMdlj5, tg, td_new, j5g, STT, 9, 0.05);
STT = computeSTT(gprMdlj6, tg, td_new, j6g, STT, 11, 0.09);
STT = computeSTT(gprMdlj7, tg, td_new, j7g, STT, 13, 0.09);

% Add time column to STT
STT = [td_new, STT];

% Get the directory where the script is located
scriptDir = fileparts(mfilename('fullpath'));

% Save STT to CSV in the same directory as the script
outputFilePath = fullfile(scriptDir, 'STT_arm_LfD_tst.csv');
writematrix(STT, outputFilePath);

disp(['STT_arm_LfD_tst.csv saved to: ', outputFilePath]);

% Plot STT for each joint
figure;
for i = 1:7
    subplot(7, 1, i);
    plot(td_new, STT(:, 2*i), 'r-', 'LineWidth', 2);
    hold on;
    plot(td_new, STT(:, 2*i+1), 'b-', 'LineWidth', 2);
    xlabel('t');
    ylabel(['j' num2str(i)]);
    legend('\mu + \rho', '\mu - \rho');
end