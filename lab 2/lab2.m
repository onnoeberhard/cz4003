%% 1.a Load image, transform to greyscale and display
img = imread('maccropped.jpg');
img = im2double(rgb2gray(img));

figure
imshow(img)

%% 1.b Sobel edge detection
h = [1 2 1; 0 0 0; -1 -2 -1];
v = [1 0 -1; 2 0 -2; 1 0 -1];

edges_h = conv2(img, h);
edges_v = conv2(img, v);

figure
imagesc(edges_h)    % imshow will discard negative values
figure
imagesc(edges_v)

% Diagonal edges mostly get separated into horizontal and vertical
% components, like the edge of what looks like a pavilion in the right
% of the image or the tip of the big tree in the front.

%% 1.c Combining edges
edges = edges_h.^2 + edges_v.^2;
figure
imshow(edges)

% Getting rid of the edges of the image and normalising
edges = edges(3:end-2, 3:end-2);
edges = edges / max(max(edges));   
figure
imshow(edges)

% The squaring will dismiss values that are very small (insignificant) in
% both filtered images. It will also get rid of the minus sign resulting
% from Sobel filtering.

%% 1.d Thresholding
img_t1 = edges > .1;
img_t2 = edges > .3;
img_t3 = edges > .5;

figure
imshow(img_t1)
figure
imshow(img_t2)
figure
imshow(img_t3)

% An advantage of thresholding is that it is very simple, but still often
% produces acceptable results. It is often important to have a binary image
% (e.g. as a mask), so tresholding (or similar techniques) are necessary
% tools in that context.
% The drawback of thresholding is that it is a very naive technique. Every
% pixel is processed in the same way and independent of its surroundings.
% This results in the breaking up of (obvious) edges among other things.

%% 1.e Canny edge detection
tl = 0.04;
th = 0.1;
sigma = 1;
E = edge(img, 'canny', [tl th], sigma);
figure
imshow(E)

sigma = 2.5;
E = edge(img, 'canny', [tl th], sigma);
figure
imshow(E)

sigma = 5;
E = edge(img, 'canny', [tl th], sigma);
figure
imshow(E)

sigma = 1; tl = 0.09;
E = edge(img, 'canny', [tl th], sigma);
figure
imshow(E)

tl = 0;
E = edge(img, 'canny', [tl th], sigma);
figure
imshow(E)

% A lower sigma value corresponds to more noisy, but also more accurate
% edges. A higher sigma results in fewer and shorter edges, but these are
% far less accurate. In the sigma = 5 image most shapes have taken the form
% of "blobs", where no persons or trees can be distinguished.

% Changing the threshold value results in more (lower tl) or less (higher
% tl) edgels detected.

%% 2.a Canny Edge Detection
tl = 0.04;
th = 0.1;
sigma = 1;
E = edge(img, 'canny', [tl th], sigma);
figure
imshow(E)

%% 2.b Hough Transform
[R, xp] = radon(E);
figure
imagesc(0:179, xp, R)

%% 2.c Find best line
[i, j] = find(R == max(R(:)));
radius = xp(i);
theta = j * pi / 180;

%% 2.d Transform coordinates
[x0, y0] = pol2cart(theta, radius);
y = @(x) size(img, 1) / 2 - (x0^2 + y0^2 - x0*(x - size(img, 2)/2)) / y0;

%% 2.f Display image and line
figure
imshow(img)
hold on
x = linspace(0, size(img, 2));
plot(x, y(x), 'r')

%% 3.b Load images
Il = im2double(rgb2gray(imread('corridorl.jpg')));
Ir = im2double(rgb2gray(imread('corridorr.jpg')));
figure
imshow(Il)
figure
imshow(Ir)
%% 2.c Execute algorithm
d = dmap(Il, Ir, 11, 11);
figure
imagesc(d, [-15 15])
%% 2.d Using real images
Il = im2double(rgb2gray(imread('triclopsi2l.jpg')));
Ir = im2double(rgb2gray(imread('triclopsi2r.jpg')));
figure
imshow(Il)
figure
imshow(Ir)
d = dmap(Il, Ir, 11, 11);
figure
imagesc(d, [-15 15])