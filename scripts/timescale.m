clc;
clear;

STT = readmatrix('STT_arm_LfD_tst.csv');

% Extract columns from STT
t1 = STT(:, 1);
gamJ1_L = STT(:, 2);
gamJ2_L = STT(:, 4);
gamJ3_L = STT(:, 6);
gamJ4_L = STT(:, 8);
gamJ5_L = STT(:, 10);
gamJ6_L = STT(:, 12);
gamJ7_L = STT(:, 14);
gamJ1_U = STT(:, 3);
gamJ2_U = STT(:, 5);
gamJ3_U = STT(:, 7);
gamJ4_U = STT(:, 9);
gamJ5_U = STT(:, 11);
gamJ6_U = STT(:, 13);
gamJ7_U = STT(:, 15);

tscale = t1/0.8;

STT(:,1) = tscale;

% Get the directory where the script is located
scriptDir = fileparts(mfilename('fullpath'));

% Save STT to CSV in the same directory as the script
outputFilePath = fullfile(scriptDir, 'STT_arm_LfD_tst_scaled.csv');
writematrix(STT, outputFilePath);


% Plot J1
subplot(7, 1, 1);
plot(tscale, gamJ1_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ1_U, 'b-', 'LineWidth', 2);
% plot(t2, j1, 'k-', 'LineWidth', 2);
ylabel('J1');
legend('Lower Bound', 'Upper Bound', 'Tracked Data');
title('Joint Trajectories');

% Plot J2
subplot(7, 1, 2);
plot(tscale, gamJ2_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ2_U, 'b-', 'LineWidth', 2);
% plot(t2, j2, 'k-', 'LineWidth', 2);
ylabel('J2');

% Plot J3
subplot(7, 1, 3);
plot(tscale, gamJ3_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ3_U, 'b-', 'LineWidth', 2);
% plot(t2, j3, 'k-', 'LineWidth', 2);
ylabel('J3');

% Plot J4
subplot(7, 1, 4);
plot(tscale, gamJ4_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ4_U, 'b-', 'LineWidth', 2);
% plot(t2, j4, 'k-', 'LineWidth', 2);
ylabel('J4');

% Plot J5
subplot(7, 1, 5);
plot(tscale, gamJ5_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ5_U, 'b-', 'LineWidth', 2);
% plot(t2, j5, 'k-', 'LineWidth', 2);
ylabel('J5');

% Plot J6
subplot(7, 1, 6);
plot(tscale, gamJ6_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ6_U, 'b-', 'LineWidth', 2);
% plot(t2, j6, 'k-', 'LineWidth', 2);
ylabel('J6');

% Plot J7
subplot(7, 1, 7);
plot(tscale, gamJ7_L, 'r-', 'LineWidth', 2);
hold on;
plot(tscale, gamJ7_U, 'b-', 'LineWidth', 2);
% plot(t2, j7, 'k-', 'LineWidth', 2);
xlabel('t');
ylabel('J7');

% Plot J1
subplot(7, 1, 1);
plot(t1, gamJ1_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ1_U, 'b:', 'LineWidth', 2);
% plot(t2, j1, 'k:', 'LineWidth', 2);
ylabel('J1');
legend('Lower Bound', 'Upper Bound', 'Tracked Data');
title('Joint Trajectories');

% Plot J2
subplot(7, 1, 2);
plot(t1, gamJ2_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ2_U, 'b:', 'LineWidth', 2);
% plot(t2, j2, 'k:', 'LineWidth', 2);
ylabel('J2');

% Plot J3
subplot(7, 1, 3);
plot(t1, gamJ3_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ3_U, 'b:', 'LineWidth', 2);
% plot(t2, j3, 'k:', 'LineWidth', 2);
ylabel('J3');

% Plot J4
subplot(7, 1, 4);
plot(t1, gamJ4_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ4_U, 'b:', 'LineWidth', 2);
% plot(t2, j4, 'k:', 'LineWidth', 2);
ylabel('J4');

% Plot J5
subplot(7, 1, 5);
plot(t1, gamJ5_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ5_U, 'b:', 'LineWidth', 2);
% plot(t2, j5, 'k:', 'LineWidth', 2);
ylabel('J5');

% Plot J6
subplot(7, 1, 6);
plot(t1, gamJ6_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ6_U, 'b:', 'LineWidth', 2);
% plot(t2, j6, 'k:', 'LineWidth', 2);
ylabel('J6');

% Plot J7
subplot(7, 1, 7);
plot(t1, gamJ7_L, 'r:', 'LineWidth', 2);
hold on;
plot(t1, gamJ7_U, 'b:', 'LineWidth', 2);
% plot(t2, j7, 'k:', 'LineWidth', 2);
xlabel('t');
ylabel('J7');