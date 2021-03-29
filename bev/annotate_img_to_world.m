%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% calibrate img to world
%%%% 2020/12/24
%%%%%%%%%%%%%%%%%%%%%%%%%%

path_img = '/home/minghanz/Pictures/vlcsnap-2020-12-24-16h39m07s310.png';
path_bev = '/home/minghanz/Pictures/jackson_town_square_bev2.jpg';

%%%%% annotate on img
im_img = imread(path_img);
image(im_img);
axis image
hold on

pts_img = ginput();

if ~ isempty(pts_img)
    plot(pts_img(:,1), pts_img(:,2), 'rx');
end

figure();

%%%%% annotate on bev
im_bev = imread(path_bev);
image(im_bev);
axis image
hold on

pts = ginput();

% pts = [1575, 611; 1428, 608; 1256, 605; 1066, 876;
%                         1368, 924; 1866, 601];
% % pts = pts + 1;
% pts(:,1) = pts(:,1) / 1920 * 852;
% pts(:,2) = pts(:,2) / 1080 * 480;

if ~ isempty(pts)
    plot(pts(:,1), pts(:,2), 'rx');
end

pts_ruler = ginput();

if ~ isempty(pts_ruler)
    plot(pts_ruler(:,1), pts_ruler(:,2), 'gx');
end
px_per_m = (pts_ruler(2,1)-pts_ruler(1,1)) / 10;
pts_0 = pts(1, :);
pts_rel = pts - pts_0;
pts_world = pts_rel / px_per_m;