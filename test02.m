close all
clear

% Step01-Data loading and pre-processing

% Set images and labels path
imDir = fullfile('data_for_moodle','images_256')
pxDir = fullfile('data_for_moodle','labels_256')
% labelsWithTwo: Stores a new folder that turns 5 labels into 2 labels
pxcDir = fullfile('data_for_moodle','labelsWithTwo')

% step01-01 convert 5 labels images into 2 labels images
labelImages = dir(fullfile(pxDir,'*.png')) % 'dir' get the information about label images
% 
% for i = 1:length(labelImages)
%     [labelImage,map] = imread(fullfile(pxDir, labelImages(i).name));
%     labelImage(labelImage ~= 1) = 3 % change the pixel do not equil 1 to 3 (include:0,2,3,4)
%     imwrite(labelImage,map,fullfile(pxcDir,labelImages(i).name)); % save the changed images to labelsWithTwo好了
% end
% Note: map is a color mapping table (if the image is an indexed color image). 
% If the image is a standard RGB or grayscale image, the map will be empty.

% step01-02 Screen out the images with labels

% Create an image data store and lable store
imds = imageDatastore(imDir);
pxds = imageDatastore(pxcDir);

% Extract the filenames both images and labels without extensions
% Gets the absolute path to the file
imDir_ab = fullfile(pwd, imDir);
pxcDir_ab = fullfile(pwd, pxcDir);

imageFileNames = cellfun(@(x) erase(extractBefore(x, '.'), imDir_ab), imds.Files, 'UniformOutput', false);

labelFileNames = cellfun(@(x) erase(extractBefore(x, '.'), pxcDir_ab), pxds.Files, 'UniformOutput', false);
 
% Find and remove images without labels
filesToRemove = setdiff(imageFileNames, labelFileNames);
imds.Files(ismember(imageFileNames, filesToRemove)) = [];

% step02-prepare to the trainning data and validation data

% step02-01 define classes and label
numClasses = 2;
 
classNames = ["flower",'background'];
pixelLabelIDs = [1,3];
 
pxds = pixelLabelDatastore(pxcDir,classNames,pixelLabelIDs); % To ensure a one-to-one correspondence
% In this tast, this step could be omitted, because the pxds has already correspondence

% step02-02 Split the traindata and validationdata
numTrainFiles = 700; % It has 846 files, to ensure the validation, I choice nearly 150files to validate
 
numFiles = numel(imds.Files); % calculate the number of the files
indices = randperm(numFiles); % generate randomly arranged indexes
 
trainIndices = indices(1:numTrainFiles); % 1-700
valIndices = indices(numTrainFiles+1:end); % +1 because the indices start with '0'


% subset the images and pixel stores by indices
imdsTrain = subset(imds, trainIndices);
imdsValidation = subset(imds, valIndices);
pxdsTrain = subset(pxds, trainIndices);
pxdsValidation = subset(pxds, valIndices);

trainingData = combine(imdsTrain, pxdsTrain);
validationData = combine(imdsValidation, pxdsValidation);

 
%  calculate the weight

tb1 = countEachLabel(pxds)

totalNumberOfPixels = sum(tb1.PixelCount); % Calculate the total number of pixels for all classes
frequency = tb1.PixelCount / totalNumberOfPixels; % Calculate the frequency of each category of pixels
classWeights = 1./frequency  % calculate the weight, 1.means 1

% step03-Build the U-net
numFilters = 64;
filterSize = 3;
 
layers = [
    % downsampling layers
    imageInputLayer([256 256 3])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
 
    % upsampling layers
    transposedConv2dLayer(4,numFilters,"Stride",2,"Cropping",1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer('Classes',tb1.Name,'ClassWeights',classWeights);
    ]
 
 
opts = trainingOptions("sgdm", ...
    'InitialLearnRate',1e-3,...
    'MaxEpochs',4,...
    'MiniBatchSize',64, ...
    'Plots','training-progress');

% step04-Train the net
net = trainNetwork(trainingData, layers, opts)

% step05-Test the net 
% classname = categorical(imdsValidation)
% scores = minibatchqueue(net,imdsValidation); % The probability corresponding to each label
% YValidation = scores2label(scores,classNames); % The label with the highest matching probability
% 
% TValidation = imdsValidation.Labels;
% accurcy = mean(YValidation == TValidation)

% predictedLabels = semanticseg(imdsValidation,net);
% predictedLabels = categorical(predictedLabels);
% expectedLables = pxdsValidation.readall();
% 
% if iscell(expectedLables)
%     expectedLables = cat(3, expectedLables{:});
% end
% 
% if -iscategorical(expectedLables)
%     expectedLables = categorical(expectedLables);
% end
% 
% accuracy = mean(predictedLabels == expectedLables, 'all')