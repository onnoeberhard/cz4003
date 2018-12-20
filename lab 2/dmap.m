function d = dmap(Il, Ir, th, tw)

[h, w] = size(Il);
d = zeros(h, w);

th_ = floor(th / 2);
tw_ = floor(tw / 2);

for y = th_+1 : h-th_
    for x = tw_+1 : w-tw_
        T = rot90(Il(y-th_ : y+th_, x-tw_ : x+tw_), 2);
        S = conv2(Ir(y-th_ : y+th_, :).^2, ones(th, tw), 'same') ...
            - 2*conv2(Ir(y-th_ : y+th_, :), T, 'same') ...
            + sum(sum(T.^2));
        xr = find(S(tw_+1, :) == min(S(tw_+1, :)), 1);
        d(y, x) = x - xr;
    end
end