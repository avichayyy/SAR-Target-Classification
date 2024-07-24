% outputFolder = pwd;
% dataURL = ['https://ssd.mathworks.com/supportfiles/radar/data/' ...
%     'MSTAR_TargetData.tar.gz'];
% helperDownloadMSTARTargetData(outputFolder,dataURL);
% 
% function helperDownloadMSTARTargetData(outputFolder,DataURL)
% % Download the data set from the given URL to the output folder.
% 
%     radarDataTarFile = fullfile(outputFolder,'MSTAR_TargetData.tar.gz');
%     
%     if ~exist(radarDataTarFile,'file')
%         
%         disp('Downloading MSTAR Target data (28 MiB)...');
%         websave(radarDataTarFile,DataURL);
%         untar(radarDataTarFile,outputFolder);
%     end
% end
sarDatasetPath = fullfile(pwd,'Data');
imds = imageDatastore(sarDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
rng(0)
figure
% Shuffle the datastore.
imds = shuffle(imds);
for i = 1:20
    subplot(4,5,i)
    img = read(imds);
    imshow(img)
    title(imds.Labels(i))
    sgtitle('Sample training images')
end
% Count labels
labelCount = countEachLabel(imds);

% Define Image size
imgSize = [128,128,1];

% Create Datastore Object for Training, Validation and Testing
trainingPct = 0.8;
validationPct = 0.1;
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,...
    trainingPct,validationPct,'randomize');

% Data Augmentation
auimdsTrain = augmentedImageDatastore(imgSize, imdsTrain);
auimdsValidation = augmentedImageDatastore(imgSize, imdsValidation);
auimdsTest = augmentedImageDatastore(imgSize, imdsTest);

% Define Network Architecture
layers = createNetwork(imgSize);

% Train Network
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',3, ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize',48,...
    'ValidationData',auimdsValidation, ...
    'ValidationFrequency',15, ...
    'Verbose',false, ...
    'CheckpointPath',tempdir,...
    'Plots','training-progress');

net = trainNetwork(auimdsTrain,layers,options);

% Classify Test images and calc Accuracy
YPred = classify(net,auimdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

figure
cm = confusionchart(YPred, YTest);
cm.RowSummary = 'row-normalized';
cm.Title = 'SAR Target Classification Confusion Matrix';

function layers = createNetwork(imgSize)
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1])      % Input Layer
        convolution2dLayer(3,32,'Padding','same')       % Convolution Layer
        reluLayer                                       % Relu Layer
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer                         % Batch normalization Layer
        reluLayer
                                                        % 128x128x32
        maxPooling2dLayer(2,'Stride',2)                 % Max Pooling Layer - 64x64x32
        
        convolution2dLayer(3,64,'Padding','same')       % conv filters - (32x3x3)x 64 Filters
        reluLayer
        convolution2dLayer(3,64,'Padding','same')       % 64x64x64 image size
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)                 % output image - 32x32x64
        
        convolution2dLayer(3,128,'Padding','same')      % conv filters - (64x3x3)x 128 Filters
        reluLayer
        convolution2dLayer(3,128,'Padding','same')      % 32x32x128 image size
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)                 % output image - 16x16x128 
    
        convolution2dLayer(3,256,'Padding','same')
        reluLayer
        convolution2dLayer(3,256,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)                 % output image - 8x8x256            
    
        convolution2dLayer(6,512)                       % output image - 1x1x512
        reluLayer
        
        dropoutLayer(0.5)                               % Dropout Layer - Very high odd
        fullyConnectedLayer(512)                        % Fully connected Layer.
        reluLayer
        fullyConnectedLayer(8)
        softmaxLayer                                    % Softmax Layer
        classificationLayer                             % Classification Layer
        ];
end