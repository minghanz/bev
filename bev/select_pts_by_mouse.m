%%% pick points on image by mouse and get their coordinates

% path = '/media/sda1/datasets/extracted/KoPER/added/SK_4_empty_road_bev.png';
% path = '/media/sda1/datasets/extracted/roadmanship_format/CARLA/trial2/images/ori/0000000293.jpg';
% path = '/home/minghanz/Pictures/vlcsnap-2020-08-17-16h30m32s275.png';
% path = '/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/07/outputs_gen_CARLA_half_angle_shadow/videos/bev_snap.png';
% im = imread('/media/sda1/datasets/extracted/KoPER/Sequence1a/aaa/KAB_SK_1_undist_1384779397760009.bmp');
% im = imread('/media//sda1/datasets/extracted/KoPER/added/Sequence1a/4/KAB_SK_4_undist_1384779316559997.bmp');
% path = '/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/videos/vlcsnap-2020-10-27-16h51m56s699.png';
% path = '/home/minghanz/Pictures/vlcsnap-2020-12-01-21h40m09s656.png';
% path = '/home/minghanz/Pictures/vlcsnap-2020-12-24-15h06m21s791.png';
% %%% bev
% path = '/home/minghanz/Pictures/jackson_town_square_bev2.jpg';
% path = '/home/minghanz/Pictures/new_bev/bev_screenshot_24.12.20204.png';
% path = '/home/minghanz/Pictures/vlcsnap-2021-03-05-00h20m25s387.png';
path = '/home/minghanz/Pictures/vlcsnap-2021-03-04-19h56m15s184.png';
%%% ori
% path = '/home/minghanz/Pictures/vlcsnap-2020-12-24-15h06m21s791.png'; % 2020, 480p
% path = '/home/minghanz/Pictures/vlcsnap-2020-08-17-17h50m57s450.png';
% path = '/home/minghanz/Pictures/vlcsnap-2020-12-24-16h39m07s310.png';
im = imread(path);
image(im);
axis image
hold on

% pts_img = ginput();
% 
% if ~ isempty(pts_img)
%     plot(pts_img(:,1), pts_img(:,2), 'rx');
% end
% 
pts = ginput();

% pts = [1575, 611; 1428, 608; 1256, 605; 1066, 876;
%                         1368, 924; 1866, 601];
% % pts = pts + 1;
% pts(:,1) = pts(:,1) / 1920 * 852;
% pts(:,2) = pts(:,2) / 1080 * 480;

if ~ isempty(pts)
    plot(pts(:,1), pts(:,2), 'rx');
end