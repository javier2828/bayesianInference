    % Credits: mathematicalmonk: mathematicalmonk's channel
    % Choose a kernel (covariance function) 
    kernel = 3; 
     
    switch kernel
        case 1; k = @(x,y) 1*x'*y; % Linear 
        case 2; k = @(x,y) 1*min(x,y); % Brownian
        case 3; k = @(x,y) exp(-100*(x-y)'*(x-y)) % Squared exponential
        case 4; k = @(x,y) exp(-1*sqrt((x-y)'*(x-y)))
        case 5; k = @(x,y) exp(-1*sin(50*pi*(x-y))^2) % Periodic
        case 6; k = @(x,y) exp(-100*min(abs(x-y), abs(x+y))^2)
    end  
            
    % Choose points at which to sample 15
    x= (0:.01:1); 
    n = length(x); 
     
    % Construct the covariance matrix 
    C = zeros(n,n);
    for i = 1:n 
        for j = 1:n 
            C(i,j)= k(x(i),x(j));
        end
    end
     
    % Sample from the Gaussian process at these points
    u = randn(n,1); % sample u ~ N(0, I)
    [A, S, B] = svd(C); % factor C = ASB'. Possible because C is symmetric. Notice in fact C = ASA' = (A * (sqrt(S))) * (A *(sqrt(S)))'
    % See PP 6.6 minute 6:11: Fact a: 
    z = A*sqrt(S)*u; % z = A S^.5 u 
     
    % Plot 
    figure(2); hold on; clf
    plot(x,z,'.-')
    axis([0, 1, -2, 2])