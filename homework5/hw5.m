clear;
clc;

trainingData = load("hw5Data/TrainingSamplesDCT_8_new.mat");
foreground = trainingData.TrainsampleDCT_FG;
background = trainingData.TrainsampleDCT_BG;

[row_fg, col_fg] = size(foreground);
[row_bg, col_bg] = size(background);
priorYCheetah = row_fg / (row_bg + row_fg);
priorYGrass = row_bg / (row_bg + row_fg);

zigzagPattern = load('Zig-Zag Pattern.txt');
zigzagPattern = zigzagPattern + 1; % 1 indexing in MATLAB

original_Image = imread('cheetah.bmp');
pad_Image = padarray(original_Image, [7 7], 'replicate', 'pre');
imageModified = im2double(pad_Image);
[image_row, image_col] = size(imageModified);

A = zeros(image_row, image_col, 64);
for i = 1:image_row - 7
    for j = 1:image_col - 7
        block = imageModified(i:i+7, j: j+7);
        dctOutput = dct2(block);
        orderedDCTOutput(zigzagPattern(:)) = dctOutput(:);
        A(i, j, :) = orderedDCTOutput;

    end
end


groundTruth = imread('cheetah_mask.bmp');
groundTruthModified = im2double(groundTruth);

groundTruthFGCount = 0;
groundTruthBGCount = 0;
for i = 1 : image_row - 7
    for j = 1 : image_col - 7
        if groundTruthModified(i, j) == 1
            groundTruthFGCount = groundTruthFGCount + 1;
        else 
            groundTruthBGCount = groundTruthBGCount + 1;
        end
    end
end

% a) part
dimensions = 64;
c = 8;

mu_fg = cell(5, 1);
sigma_fg = cell(5, 1);
pi_fg = cell(5, 1);
 
mu_bg = cell(5, 1);
sigma_bg = cell(5, 1);
pi_bg = cell(5, 1);
for iteration = 1:5
    mu_init = rand(c, dimensions); % number of mixture components x dimensions of each gaussian
    pi_init = rand(1, c);
    pi_init = pi_init ./ sum(pi_init);
    sigma_init = rand(c, dimensions);
    sigma_init(sigma_init < 0.001) = 0.001;
    [mu_current, sigma_current, pi_current] = EM(mu_init, sigma_init, pi_init, foreground, 100, c, row_fg);
    mu_fg{iteration} = mu_current;
    sigma_fg{iteration} = sigma_current;
    pi_fg{iteration} = pi_current;
end
for iteration = 1:5
    mu_init = rand(c, dimensions);
    pi_init = rand(1, c);
    pi_init = pi_init ./ sum(pi_init);
    sigma_init = rand(c, dimensions);
    sigma_init(sigma_init < 1e-3) = 1e-3;
    [mu_current, sigma_current, pi_current] = EM(mu_init, sigma_init, pi_init, background, 100, c, row_bg);
    mu_bg{iteration} = mu_current;
    sigma_bg{iteration} = sigma_current;
    pi_bg{iteration} = pi_current;
end
save('fg_GMM_params.mat', 'mu_fg', 'sigma_fg', 'pi_fg');
save('bg_GMM_params.mat', 'mu_bg', 'sigma_bg', 'pi_bg');
errors = zeros(5,5,11);
dims = [1,2,4,8,16,24,32,40,48,56,64];
for k = 1:length(dims)
    for n1 = 1:5
        for n2 = 1:5
            calculatedMask = zeros(image_row - 7, image_col - 7);
            for i = 1:image_row - 7
                for j = 1:image_col - 7
                    dctoutput_dims = reshape(A(i, j, 1:dims(k)), [1 dims(k)]);
                    try
                        bgProb = 0;
                        for subclass = 1:c
                            bgProb = bgProb + (calculateGaussian(dctoutput_dims, mu_bg{n2}(subclass, 1:dims(k)), diag(sigma_bg{n2}(subclass, 1:dims(k))), dims(k)) * pi_bg{n2}(subclass));
                        end
 
                        fgProb = 0;
                        for subclass = 1:c
                            fgProb = fgProb + (calculateGaussian(dctoutput_dims, mu_fg{n1}(subclass, 1:dims(k)), diag(sigma_fg{n1}(subclass, 1:dims(k))), dims(k)) * pi_fg{n1}(subclass));
                        end
                    catch ME
                        disp(n1);
                        disp(n2);
                        disp(k);
                        disp(ME.identifier);
                    end
                    if(fgProb*priorYCheetah > bgProb*priorYGrass)
                        calculatedMask(i,j) = 1;
                    else
                        calculatedMask(i,j) = 0;
                    end
                end
            end
            errors(n1, n2, k) = calculateErrorCount(groundTruthModified, calculatedMask, image_row-7, image_col-7, groundTruthFGCount, groundTruthBGCount, priorYCheetah, priorYGrass);
        end
    end
end
save('errors.mat','errors');
x_axis = [1,2,4,8,16,24,32,40,48,56,64];
for n1 = 1:5 % foreground
    figure;
    for n2 = 1:5 % background
        temp_error = reshape(errors(n1,n2,:), [1 11]);
        plot(x_axis, temp_error);
        hold on
    end
    grid on
    legend('BG Mixture 1', 'BG Mixture 2', 'BG Mixture 3', 'BG Mixture 4','BG Mixture 5');
    titleText = ['Probability of error vs dimensions for foreground mixture: ', num2str(n1)];
    title(titleText);
    xlabel('number of dimensions');
    ylabel('probability of error');
    hold off
end

% b) part
new_C = [1,2,4,8,16,32];
mu_fg_params = cell(6,1);
sigma_fg_params = cell(6, 1);
pi_fg_params = cell(6, 1);

mu_bg_params = cell(6, 1);
sigma_bg_params = cell(6, 1);
pi_bg_params = cell(6, 1);

for i = 1:length(new_C)
    mu_init = rand(new_C(i), dimensions); % number of mixture components x dimensions of each gaussian
    pi_init = rand(1, new_C(i));
    pi_init = pi_init / sum(pi_init);
    sigma_init = rand(new_C(i), dimensions);
    sigma_init(sigma_init < 0.001) = 0.001;
    [mu_current, sigma_current, pi_current] = EM(mu_init, sigma_init, pi_init, foreground, 100, new_C(i), row_fg);
    mu_fg_params{i} = mu_current;
    sigma_fg_params{i} = sigma_current;
    pi_fg_params{i} = pi_current;
end
for i=1:length(new_C)
    mu_init = rand(new_C(i), dimensions); % number of mixture components x dimensions of each gaussian
    pi_init = rand(1, new_C(i));
    pi_init = pi_init ./ sum(pi_init);
    sigma_init = rand(new_C(i), dimensions);
    sigma_init(sigma_init < 0.001) = 0.001;
    [mu_current, sigma_current, pi_current] = EM(mu_init, sigma_init, pi_init, background, 100, new_C(i), row_bg);
    mu_bg_params{i} = mu_current;
    sigma_bg_params{i} = sigma_current;
    pi_bg_params{i} = pi_current;
end
save('fg_GMM_params_partb.mat', 'mu_fg_params', 'sigma_fg_params', 'pi_fg_params');
save('bg_GMM_params_partb.mat', 'mu_bg_params', 'sigma_bg_params', 'pi_bg_params');

errors = zeros(6, 11);
dims = [1,2,4,8,16,24,32,40,48,56,64];
for i = 1:length(new_C)
    for j = 1:length(dims)
        calculatedMask = zeros(image_row - 7, image_col - 7);
        for k = 1:image_row - 7
            for l = 1:image_col - 7
                dctoutput_dims = reshape(A(k, l, 1:dims(j)), [1 dims(j)]);
                bgProb = 0;
                for subclass = 1:new_C(i)
                    bgProb = bgProb + (calculateGaussian(dctoutput_dims, mu_bg_params{i}(subclass, 1:dims(j)), diag(sigma_bg_params{i}(subclass, 1:dims(j))), dims(j)) * pi_bg_params{i}(subclass));
                end

                fgProb = 0;
                for subclass = 1:new_C(i)
                    fgProb = fgProb + (calculateGaussian(dctoutput_dims, mu_fg_params{i}(subclass, 1:dims(j)), diag(sigma_fg_params{i}(subclass, 1:dims(j))), dims(j)) * pi_fg_params{i}(subclass));
                end
                if(fgProb*priorYCheetah > bgProb*priorYGrass)
                    calculatedMask(k,l) = 1;
                else
                    calculatedMask(k,l) = 0;
                end
            end
        end
        errors(i, j) = calculateErrorCount(groundTruthModified, calculatedMask, image_row-7, image_col-7, groundTruthFGCount, groundTruthBGCount, priorYCheetah, priorYGrass);
        figure;
        imagesc(calculatedMask);
        title('Prediction');
        colormap(gray(255));
    end
end
save('newErrors.mat', 'errors');

x_axis = [1,2,4,8,16,24,32,40,48,56,64];
figure;
for i = 1:length(new_C)
    temp_error = reshape(errors(i, :), [1 11]);
    plot(x_axis, temp_error);
    hold on
    grid on
end
legend('C = 1', 'C = 2', 'C = 4', 'C = 8','C = 16', 'C = 32');
ylim([0.04 0.09]);
title('Probability of error vs dimensions');
xlabel('number of dimensions');
ylabel('probability of error');

function probError = calculateErrorCount(groundTruthModified, mask, image_row, image_col, groundTruthFGCount, groundTruthBGCount, priorCheetah, priorGrass)
    errorFGCount = 0; % false negative
    errorBGCount = 0; % false positive
    for i = 1:image_row
        for j = 1:image_col
            if mask(i,j) ==0 && groundTruthModified(i, j) == 1
                errorFGCount = errorFGCount + 1;
            elseif mask(i,j) == 1 && groundTruthModified(i, j) == 0
                errorBGCount = errorBGCount + 1;
            end
        end
    end

    fgError = errorFGCount / groundTruthFGCount;
    bgError = errorBGCount / groundTruthBGCount;

    probError = (fgError * priorCheetah) + (bgError * priorGrass);
end

function [mu_final, sigma_final, pi_final] = EM(mu_old, sigma_old, pi_old, data, iterations, c, row_fg)
    mu_current = mu_old;
    sigma_current = sigma_old;
    pi_current = pi_old;
    h = zeros(row_fg, c);
    for iter = 1: iterations
        % E step
        for i = 1:row_fg
            for j = 1:c
                % getting values of zj for each datapoint, figuring out the
                % probability of it in a mixture component
                h(i, j) = pi_old(j) * calculateGaussian(data(i, :), mu_old(j, :), diag(sigma_old(j, :)), 64);
            end
            h(i, :) = h(i, :) / sum(h(i, :));
        end

        % M step
        for j = 1:c
            colSum = sum(h(:,j)); % sigma over i hij
            pi_current(j) = (colSum) / row_fg;
            mu_current(j, :) = transpose(h(:, j)) * data ./ (colSum);
            numerator = 0;
            for i = 1:row_fg
                numerator = numerator + ((h(i,j) * (data(i, :) - mu_current(j, :)).^2));
            end
            sigma_current(j, :) = numerator / (colSum);
            sigma_current(sigma_current < 1e-3) = 1e-3; % empty bin problem
            
        end
        if(all(abs(mu_current - mu_old)./abs(mu_old) < 1e-4))
            disp(iter)
            break;
        end
        pi_old = pi_current;
        mu_old = mu_current;
        sigma_old = sigma_current;
    end
    mu_final = mu_old;
    sigma_final = sigma_old;
    pi_final = pi_old;
end


function Y = calculateGaussian(x, mu, sigma, dims)
    covdet = det(sigma);
    denom = sqrt(((2 * pi)^dims) * covdet);
    numerator = exp((-1/2) .* ((x - mu) * inv(sigma) * transpose(x - mu)));
    Y = numerator / denom;
end