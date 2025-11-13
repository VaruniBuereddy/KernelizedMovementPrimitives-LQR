% Clear workspace and close all figures
clear; clc; close all;

% Read data from CSV files
STT = readmatrix('STT_arm_LfD_tst.csv');
A = readmatrix('joint_positions_demo_tst.csv');
TD = readmatrix('tau_d_calc_record.csv');
PDT = readmatrix('end_effector_positions.csv');

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

dt = A(:,1);

dj1 = A(:, 2);
dj2 = A(:, 3);
dj3 = A(:, 4);
dj4 = A(:, 5);
dj5 = A(:, 6);
dj6 = A(:, 7);
dj7 = A(:, 8);

xd = A(:, 9);
yd = A(:, 10);
zd = A(:, 11);

t2 = TD(:, 1);

jt1 = TD(:, 2);
jt2 = TD(:, 3);
jt3 = TD(:, 4);
jt4 = TD(:, 5);
jt5 = TD(:, 6);
jt6 = TD(:, 7);
jt7 = TD(:, 8);

jp1 = TD(:, 9);
jp2 = TD(:, 10);
jp3 = TD(:, 11);
jp4 = TD(:, 12);
jp5 = TD(:, 13);
jp6 = TD(:, 14);
jp7 = TD(:, 15);

eet = PDT(:,1);

eepx = PDT(:,2);
eepy = PDT(:,3);
eepz = PDT(:,4);

figure
plot3(xd,yd,zd, 'b--', 'LineWidth',1.5); hold on;
plot3(eepx,eepy,eepz, 'r-', 'LineWidth',2.5); hold on;
grid on
axis square
xlabel('x')
ylabel('y')
zlabel('z')
axis equal

% Plot tubess with learnt trajectory
figure;

% Plot jp1
subplot(7, 1, 1);
plot(t1, gamJ1_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ1_U, 'b-', 'LineWidth', 2);
plot(t2, jp1, 'k-', 'LineWidth', 2);
ylabel('J1');
legend('Lower Bound', 'Upper Bound', 'Tracked Data');
title('Joint Trajectories');

% Plot jp2
subplot(7, 1, 2);
plot(t1, gamJ2_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ2_U, 'b-', 'LineWidth', 2);
plot(t2, jp2, 'k-', 'LineWidth', 2);
ylabel('J2');

% Plot jp3
subplot(7, 1, 3);
plot(t1, gamJ3_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ3_U, 'b-', 'LineWidth', 2);
plot(t2, jp3, 'k-', 'LineWidth', 2);
ylabel('J3');

% Plot jp4
subplot(7, 1, 4);
plot(t1, gamJ4_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ4_U, 'b-', 'LineWidth', 2);
plot(t2, jp4, 'k-', 'LineWidth', 2);
ylabel('J4');

% Plot jp5
subplot(7, 1, 5);
plot(t1, gamJ5_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ5_U, 'b-', 'LineWidth', 2);
plot(t2, jp5, 'k-', 'LineWidth', 2);
ylabel('J5');

% Plot jp6
subplot(7, 1, 6);
plot(t1, gamJ6_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ6_U, 'b-', 'LineWidth', 2);
plot(t2, jp6, 'k-', 'LineWidth', 2);
ylabel('J6');

% Plot jp7
subplot(7, 1, 7);
plot(t1, gamJ7_L, 'r-', 'LineWidth', 2);
hold on;
plot(t1, gamJ7_U, 'b-', 'LineWidth', 2);
plot(t2, jp7, 'k-', 'LineWidth', 2);
xlabel('t');
ylabel('J7');

% Adjust layout for better spacing
set(gcf, 'Position', [100, 100, 800, 1200]); % Adjust figure size

% Plot torques

figure;

for i=1:7
subplot(7, 1, i);
hold on;
plot(t2, TD(:,1+i), 'r-', 'LineWidth', 2);
%plot(t2, P(:,32+i), 'r-', 'LineWidth', 2);
% ylabel('U1');
end
% Adjust layout for better spacing
set(gcf, 'Position', [100, 100, 800, 1200]); % Adjust figure size