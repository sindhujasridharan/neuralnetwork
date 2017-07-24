
% set path of images 
imgPath = './train/';
parentDir = dir(imgPath);

% set the number of features required for each image
% each image is resized to 8x8 with 3 channels (RGB)
nFeatures = 8 * 8 * 3;

totalNImages = 0;
% for each sub-directory in "train" directory
for i = 1:length(parentDir)-2
    % count the number of images
    imgDir = dir([imgPath, parentDir(i+2).name, '/*.jpg']);   
    % counts the total number of images available for training
    totalNImages = totalNImages + length(imgDir);
end

% initialize a vector to store labels
label_train = zeros(totalNImages, 1);
% initialize an array of feature vectors - m * n matrix
% m -> number of images, n -> number of features
feat_train = zeros(totalNImages, nFeatures);

index = 0;
% for each sub-directory in the "train" directory
for i = 1:length(parentDir)-2
    disp(i);
    
    imgDir = dir([imgPath,parentDir(i+2).name,'/*.jpg']);
    
    % for each image in the sub-directories    
    for j = 1:length(imgDir)  
        % store the label at ith position
        index = index + 1;
        label_train(index) = i;

        % read image
        img = imread([imgPath,parentDir(i+2).name,'/',imgDir(j).name]);
        % resize image
        img = imresize(img, [8 8]);
        % vectorize the image and store as a feature vector at ith row
        feat_train(index,:) = img(:);
    end
    
end

% append corresponding label to the end of each row
feat_train = [feat_train label_train];

% write the features to a file
csvwrite('features_192.csv', feat_train)

disp('Done!');