% author: ahsanjalal

% Constants
MAIN_DIR = '~/Train_video_gmm_results_bkgRatio_07_numframe_250_ga_20_sz_200_disk/';
VIDEO_DIR = '~/Training_dataset/Videos/';
FOREGROUND_DETECTOR_PARAMS = {'NumGaussians', 20, 'NumTrainingFrames', 250, 'MinimumBackgroundRatio', 0.7};
BLOB_ANALYSIS_PARAMS = {'BoundingBoxOutputPort', true, 'AreaOutputPort', true, 'CentroidOutputPort', true, 'MinimumBlobArea', 200};
STRUCTURING_ELEMENT_OPEN = strel('disk', 3);
STRUCTURING_ELEMENT_CLOSE = strel('disk', 5);
FRAME_RESIZE = [640, 640];
FRAME_ADJUST_GAMMA = 1.5;

% Change to video directory
chdir(VIDEO_DIR);

% Get list of video files
video_name_list = dir('*.flv');

for vids = 1:length(video_name_list)
    video_name = video_name_list(vids).name;
    fprintf('Video number %d is in process: %s\n', vids, video_name);
    
    % Create output folder if it doesn't exist
    opfolder = fullfile(MAIN_DIR, video_name);
    if ~exist(opfolder, 'dir')
        mkdir(opfolder);
    end
    
    % Initialize foreground detector and blob analysis
    foregroundDetector = vision.ForegroundDetector(FOREGROUND_DETECTOR_PARAMS{:});
    blobAnalysis = vision.BlobAnalysis(BLOB_ANALYSIS_PARAMS{:});
    
    % Initialize video reader
    chdir(VIDEO_DIR);
    videoReader = vision.VideoFileReader(video_name);
    
    i = -1;
    while ~isDone(videoReader)
        i = i + 1;
        frame = step(videoReader);
        frame = preprocess_frame(frame);
        filteredForeground = detect_foreground(foregroundDetector, frame);
        [area, centroid, bbox] = step(blobAnalysis, filteredForeground);
        
        if ~isempty(bbox)
            save_frame_and_annotations(opfolder, i, frame, filteredForeground, bbox);
        else
            save_empty_annotation(opfolder, i);
        end
    end
end

function frame = preprocess_frame(frame)
    frame = imresize(frame, FRAME_RESIZE);
    frame = imadjust(frame, [], [], FRAME_ADJUST_GAMMA);
end

function filteredForeground = detect_foreground(foregroundDetector, frame)
    foreground = step(foregroundDetector, frame);
    filteredForeground = imopen(foreground, STRUCTURING_ELEMENT_OPEN);
    filteredForeground = imclose(filteredForeground, STRUCTURING_ELEMENT_CLOSE);
end

function save_frame_and_annotations(opfolder, frame_idx, frame, filteredForeground, bbox)
    opBaseFileName = sprintf('%3.3d.png', frame_idx);
    textfilename = sprintf('%3.3d.txt', frame_idx);
    opFullFileName = fullfile(opfolder, opBaseFileName);
    opFullFiletext = fullfile(opfolder, textfilename);
    
    test_image = zeros(size(filteredForeground));
    [img_height, img_width, ~] = size(frame);
    
    for d = 1:size(bbox, 1)
        x = bbox(d, 1);
        y = bbox(d, 2);
        w = bbox(d, 3);
        h = bbox(d, 4);
        
        test_image = fill_test_image(test_image, filteredForeground, x, y, w, h, img_height, img_width);
        
        x = double(x + w / 2.0) / img_width;
        y = double(y + h / 2.0) / img_height;
        w = double(w) / img_width;
        h = double(h) / img_height;
        
        fileID = fopen(opFullFiletext, 'a');
        fprintf(fileID, '%d %f %f %f %f\n', 0, x, y, w, h);
        fclose(fileID);
    end
    
    imwrite(test_image, opFullFileName, 'png');
end

function test_image = fill_test_image(test_image, filteredForeground, x, y, w, h, img_height, img_width)
    if y + h > img_height && x + w > img_width
        test_image(y:img_height, x:img_width) = filteredForeground(y:img_height, x:img_width);
    elseif y + h > img_height
        test_image(y:img_height, x:x + w) = filteredForeground(y:img_height, x:x + w);
    elseif x + w > img_width
        test_image(y:y + h, x:img_width) = filteredForeground(y:y + h, x:img_width);
    else
        test_image(y:y + h, x:x + w) = filteredForeground(y:y + h, x:x + w);
    end
end

function save_empty_annotation(opfolder, frame_idx)
    textfilename = sprintf('%3.3d.txt', frame_idx);
    opFullFiletext = fullfile(opfolder, textfilename);
    fileID = fopen(opFullFiletext, 'a');
    fclose(fileID);
end