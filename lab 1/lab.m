%% 1.a Load image and transform to greyscale
img = imread('mrt-train.jpg');
size(img)
img = rgb2gray(img);
size(img)

%% 1.b Display the image
figure()
imshow(img)

%% 1.c Find minimum and maximum intensity values (ideally 0 and 255)
r_min = double(min(img(:)))
r_max = double(max(img(:)))

%% 1.d Contrast stretching
img = uint8(255 * (double(img) - r_min) / (r_max - r_min));
assert(min(img(:)) == 0 && max(img(:)) == 255)    % Check if it worked

%% 1.e Display the enhanced image
figure()
imshow(img)

%% 2.a Intensity histograms
figure()
imhist(img, 10)
figure()
imhist(img, 256)
% The first histogram shows much less detail, it divides the 256 intensity
% levels into 10, making every bin hold on average 256/10 = 25.6 times as 
% many pixels as in the second histogram, where all 256 intensity levels
% get their own bin. An example of detail that gets lost in the first
% histogram is the spike at the intensity levels between 220 and 226.

%% 2.b Histogram equalization
img_eq = histeq(img, 256);    % 256 discrete intensity levels -> N=256
figure()
imhist(img_eq, 10)
figure()
imhist(img_eq, 256)
%% 2.c Repeat histogram equalization
img_eq2 = histeq(img_eq, 256);
figure()
imhist(img_eq2, 10)
figure()
imhist(img_eq2, 256)
% The histograms do not become more uniform. In history equalization, 
% repeated application does not lead to better results. This is because 
% bins will only be combined, and never separated in the algorithm.
% This means the optimum for this algorithm is not necessarily a completely
% flat histogram.

%% 3.a Generate the filters
h = @(sigma, x, y) 1 / (2 * pi * sigma^2) * exp(-(x.^2 + y.^2) / (2 * sigma^2));
[x, y] = meshgrid(-2:2);
h1 = h(1, x, y);
h1 = h1 / sum(h1(:));
h2 = h(2, x, y);
h2 = h2 / sum(h2(:));

figure()
mesh(x, y, h1)
figure()
mesh(x, y, h2)

%% 3.b Load and view image
img = imread('ntugn.jpg');
figure()
imshow(img)

%% 3.c Filtering the images
img_h1 = uint8(conv2(img, h1));
figure()
imshow(img_h1)

img_h2 = uint8(conv2(img, h2));
figure()
imshow(img_h2)

% The filters are not very effective at removing the noise, it is still
% clearly visible in both filtered images. Both filters do however make the
% image more blurred, the second filter (higher standard deviation) more so
% than the first one. The second filter is also better at removing the noise,
% but the price payed in blur and the fact that the noise still is not gone
% is probably too high to use the filter in this context.

%% 3.d Speckle noise
img = imread('ntusp.jpg');
figure()
imshow(img)

%% 3.e Filtering
img_h1 = uint8(conv2(img, h1));
figure()
imshow(img_h1)

img_h2 = uint8(conv2(img, h2));
figure()
imshow(img_h2)

% The filters are better at handling gaussian noise than speckle noise. For
% speckle noise a median filter is better suited.

%% 4. Median filtering gaussian noise
img = imread('ntugn.jpg');

img_h1 = uint8(medfilt2(img, [3, 3]));
figure()
imshow(img_h1)

img_h2 = uint8(medfilt2(img, [5, 5]));
figure()
imshow(img_h2)

%% 4.2 Median filtering speckle noise
img = imread('ntusp.jpg');

img_h1 = uint8(medfilt2(img, [3, 3]));
figure()
imshow(img_h1)

img_h2 = uint8(medfilt2(img, [5, 5]));
figure()
imshow(img_h2)

%% 5.a Interference patterns
img = imread('pckint.jpg');
figure()
imshow(img)

%% 5.b Compute Fourier spectrum
ft = fft2(img);
S = abs(ft).^2 / length(img);
figure()
imagesc(fftshift(log10(S)))

%% 5.c Reading coordinates
figure()
imagesc(log10(S))
x1 = 249;
y1 = 17;
x2 = 9;
y2 = 241;

%% 5.d Filtering
ft(y1-2 : y1+2, x1-2 : x1+2) = 0;
ft(y2-2 : y2+2, x2-2 : x2+2) = 0;
S = abs(ft).^2 / length(img);
figure()
imagesc(fftshift(log10(S)))

%% 5.e Inverse Fourier transform
img = uint8(ifft2(ft));
figure()
imshow(img)

ft(y1, :) = 0;
ft(y2, :) = 0;
ft(:, x1) = 0;
ft(:, x2) = 0;
S = abs(ft).^2 / length(img);
figure()
imagesc(fftshift(log10(S)))

img = uint8(ifft2(ft));
figure()
imshow(img)

%% 5.f Jailbreak
img = imread('primatecaged.jpg');
img = rgb2gray(img);
figure()
imshow(img)

ft = fft2(img);
S = abs(ft).^2 / length(img);
figure()
imagesc(fftshift(log10(S)))

x1 = 11;
y1 = 252;
x2 = 247;
y2 = 6;
x3 = 21;
y3 = 248;
x4 = 237;
y4 = 10;
ft(y1-2 : y1+2, x1-2 : x1+2) = 0;
ft(y2-2 : y2+2, x2-2 : x2+2) = 0;
ft(y3-2 : y3+2, x3-2 : x3+2) = 0;
ft(y4-2 : y4+2, x4-2 : x4+2) = 0;
S = abs(ft).^2 / length(img);
figure()
imagesc((log10(S)))
img = uint8(ifft2(ft));
figure()
imshow(img)

%% 6.a Load and view the image
img = imread('book.jpg');
figure()
imshow(img)

%% 6.b Specifying coordinates
[X, Y] = ginput(4);
x = [0 210 210 0];
y = [0 0 297 297];
%% 6.c 
v = zeros(8, 1);
A = zeros(8, 8);
for i = 1:4
    A(2*i-1 : 2*i, :) = [X(i) Y(i) 1 0 0 0 -x(i)*X(i) -x(i)*Y(i);
                         0 0 0 X(i) Y(i) 1 -y(i)*X(i) -y(i)*Y(i)];
    v(2*i-1 : 2*i) = [x(i), y(i)];
end
u = A \ v;
U = reshape([u; 1], 3, 3)'   % Print U matrix

% Verify U matrix
w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));
w

%% 6.d Warp and show image
T = maketform('projective', U');
i2 = imtransform(img, T, 'XData', [0 210], 'YData', [0 297]);
figure()
imshow(i2)