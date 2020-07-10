%%% pick points on image by mouse and get their coordinates

path = '/media/sda1/datasets/extracted/KoPER/added/SK_4_empty_road_bev.png';
% im = imread('/media/sda1/datasets/extracted/KoPER/Sequence1a/aaa/KAB_SK_1_undist_1384779397760009.bmp');
% im = imread('/media//sda1/datasets/extracted/KoPER/added/Sequence1a/4/KAB_SK_4_undist_1384779316559997.bmp');
im = imread(path);
image(im);
axis image
hold on

pts = ginput();
if ~ isempty(pts)
    plot(pts(:,1), pts(:,2), 'x');
end
