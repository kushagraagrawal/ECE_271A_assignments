clear
clc

trainSample = load('TrainingSamplesDCT_8.mat');
fgSamples = trainSample.TrainsampleDCT_FG;
bgSamples = trainSample.TrainsampleDCT_BG;

fgSamplesDim = size(fgSamples);
bgSamplesDim = size(bgSamples);

% fgSamples has 250 training examples of 64 features each
% bgSamples has 1053 training examples of 64 features each

priorYCheetah = fgSamplesDim(1) / (fgSamplesDim(1) + bgSamplesDim(1));
priorYGrass = bgSamplesDim(1) / (fgSamplesDim(1) + bgSamplesDim(1));

disp('prior probability of Cheetah');
disp(priorYCheetah);
disp('prior probability of Grass');
disp(priorYGrass);

% histogram plotting for CCD

fgScalar = zeros(fgSamplesDim(1), 1);
bgScalar = zeros(bgSamplesDim(1), 1);

for i = 1:fgSamplesDim(1)
    [value, position] = sort(abs(fgSamples(i,:)), 'descend');
    fgScalar(i) = position(2); % take position of the second highest
end

for i = 1:bgSamplesDim(1)
    [value, position] = sort(abs(bgSamples(i,:)), 'descend');
    bgScalar(i) = position(2); % take position of the second highest
end

binRange = 0.5 : 1 : 63.5;

fgCount = histcounts(fgScalar, binRange);
bgCount = histcounts(bgScalar, binRange);
fgProb = fgCount / sum(fgCount); % normalised CCD
bgProb = bgCount / sum(bgCount); % normalised CCD

figure;
h2 = histogram('BinCounts', fgProb, 'BinEdges', binRange);
xlabel('Index of 2nd Largest Coefficient')
ylabel('P(X|cheetah)')
title('Histogram of foreground')
figure;
h3 = histogram('BinCounts', bgProb, 'BinEdges', binRange);
xlabel('Index of 2nd Largest Coefficient')
ylabel('P(X|grass)')
title('Histogram of background')

original_Image = imread('cheetah.bmp');
pad_Image = padarray(original_Image, [7 7], 'replicate', 'post');
imageModified = im2double(pad_Image);
[image_row, image_col] = size(imageModified);

% create feature vector
zigzagPattern = load('Zig-Zag Pattern.txt');
zigzagPattern = zigzagPattern + 1; % 1 indexing in MATLAB

featureVector = zeros(image_row - 7, image_col - 7);
for i = 1:image_row - 7
    for j = 1:image_col - 7
        block = imageModified(i:i+7, j: j+7);
        dctOutput = dct2(block);
        orderedDCTOutput(zigzagPattern(:)) = dctOutput(:);
        [value, sortedDCTOutput] = sort(abs(orderedDCTOutput), 'descend');
%         disp(sortedDCTOutput);
        featureVector(i, j) = sortedDCTOutput(2);
    end
end

A = zeros(image_row - 7, image_col - 7);
for i = 1:image_row - 7
    for j = 1:image_col - 7
        if fgProb(1, featureVector(i, j)) * priorYCheetah > bgProb(1, featureVector(i, j)) * priorYGrass
            A(i, j) = 1;
        else
            A(i, j) = 0;
        end

    end
end

figure;
imagesc(A);
title('Prediction');
colormap(gray(255));

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

errorFGCount = 0; % false negative
errorBGCount = 0; % false positive
for i = 1:image_row - 7
    for j = 1:image_col - 7
        if A(i,j) == 0 && groundTruthModified(i, j) == 1 % P(grass | cheetah)
            errorFGCount = errorFGCount + 1;
        elseif A(i,j) == 1 && groundTruthModified(i, j) == 0 % P(cheetah | grass)
            errorBGCount = errorBGCount + 1;
        end
    end
end

fgError = errorFGCount / groundTruthFGCount;
bgError = errorBGCount / groundTruthBGCount;

probError = (fgError * priorYCheetah) + (bgError * priorYGrass);

disp('Probability of Error');
disp(probError);

