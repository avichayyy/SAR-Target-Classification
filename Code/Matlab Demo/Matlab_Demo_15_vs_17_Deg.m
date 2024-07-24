% Define dataset paths
trainDatasetPath = 'C:\Users\Avichai\Documents\MATLAB\Data10c\17_DEG';
testDatasetPath = 'C:\Users\Avichai\Documents\MATLAB\Data10c\15_DEG';

% Create image datastores
imdsTrain = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Display sample images from the training dataset
rng(9)
figure
% Shuffle the training datastore.
imdsTrain = shuffle(imdsTrain);
for i = 1:20
    subplot(4,5,i)
    img = read(imdsTrain);
    imshow(img)
    title(imdsTrain.Labels(i))
    sgtitle('Sample training images')
end

% Count labels in training and test sets
labelCountTrain = countEachLabel(imdsTrain);
labelCountTest = countEachLabel(imdsTest);

% Define image size
imgSize = [128, 128, 1];

% Data augmentation
auimdsTrain = augmentedImageDatastore(imgSize, imdsTrain);
auimdsTest = augmentedImageDatastore(imgSize, imdsTest);

% Define network architecture
layers = createNetwork(imgSize);

% Training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 3, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', 48, ...
    'Verbose', false, ...
    'CheckpointPath', tempdir, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(auimdsTrain, layers, options);

% Classify test images and calculate accuracy
YPred = classify(net, auimdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest) / numel(YTest)

% Display confusion matrix
figure
cm = confusionchart(YPred, YTest);
cm.RowSummary = 'row-normalized';
cm.Title = 'SAR Target Classification Confusion Matrix';

% Network creation function
function layers = createNetwork(imgSize)
    layers = [
        imageInputLayer([imgSize(1) imgSize(2) 1])      % Input Layer
        convolution2dLayer(3, 32, 'Padding', 'same')    % Convolution Layer
        reluLayer                                       % Relu Layer
        convolution2dLayer(3, 32, 'Padding', 'same')
        batchNormalizationLayer                         % Batch normalization Layer
        reluLayer
                                                        % 128x128x32
        maxPooling2dLayer(2, 'Stride', 2)               % Max Pooling Layer - 64x64x32
        
        convolution2dLayer(3, 64, 'Padding', 'same')    % conv filters - (32x3x3)x 64 Filters
        reluLayer
        convolution2dLayer(3, 64, 'Padding', 'same')    % 64x64x64 image size
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)               % output image - 32x32x64
        
        convolution2dLayer(3, 128, 'Padding', 'same')   % conv filters - (64x3x3)x 128 Filters
        reluLayer
        convolution2dLayer(3, 128, 'Padding', 'same')   % 32x32x128 image size
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)               % output image - 16x16x128 
    
        convolution2dLayer(3, 256, 'Padding', 'same')
        reluLayer
        convolution2dLayer(3, 256, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)               % output image - 8x8x256            
    
        convolution2dLayer(6, 512)                      % output image - 1x1x512
        reluLayer
        
        dropoutLayer(0.5)                               % Dropout Layer - Very high odd
        fullyConnectedLayer(512)                        % Fully connected Layer.
        reluLayer
        fullyConnectedLayer(10)
        softmaxLayer                                    % Softmax Layer
        classificationLayer                             % Classification Layer
        ];
end
