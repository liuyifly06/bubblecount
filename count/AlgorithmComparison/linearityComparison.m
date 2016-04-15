% comparing linearity of different method (curavture method and hough transform currently)

system('sudo python CurvatureLinearity.py 1');
hou_linearity_data = HoughLinearity();
cur_linearity_data = reshape(load('number.txt'),3,41)';
save; 
S = size(hou_linearity_data);
[a_hou, b_hou, sigma_ahou, sigma_bhou] =...
    york_fit(hou_linearity_data(:,1)', hou_linearity_data(:,2)', ...
        sqrt(hou_linearity_data(:,1))'/3, hou_linearity_data(:,3)', ...
        zeros(1,S(1)));%estimate parameters

w_hou = [b_hou, -1];
error_hou = mean(sqrt((w_hou * hou_linearity_data(:,1:2)' + a_hou).^2)); 

S = size(cur_linearity_data);    
[a_cur, b_cur, sigma_acur, sigma_bcur] =...
    york_fit(cur_linearity_data(:,1)', cur_linearity_data(:,2)', ...
        sqrt(cur_linearity_data(:,1))'/3, cur_linearity_data(:,3)', ...
        zeros(1,S(1)));%estimate parameters

w_cur = [b_cur, -1];
error_cur = mean(sqrt((w_cur * cur_linearity_data(:,1:2)' + a_cur).^2)); 

close all;    
figure('Name','Algorithm Performance Comparison','NumberTitle','off');
h=zeros(2,1);
plot(hou_linearity_data(:,1),hou_linearity_data(:,2),'b.','MarkerSize',20);
hold on
plot(cur_linearity_data(:,1),cur_linearity_data(:,2),'r.','MarkerSize',20);

%plot error ellipse for each point
for i=1:S(1)
    X(i) = hou_linearity_data(i,1);
    Y(i) = hou_linearity_data(i,2);
    sigma_X(i) =  sqrt(hou_linearity_data(i,1))/3;
    sigma_Y(i) =  hou_linearity_data(i,3);
    [Xe,Ye] = ellipse(X(i),Y(i),sigma_X(i),sigma_Y(i),32);
    plot(Xe,Ye,'color',[0.1 0.1 0.5])
    
    X(i) = cur_linearity_data(i,1);
    Y(i) = cur_linearity_data(i,2);
    sigma_X(i) =  sqrt(cur_linearity_data(i,1))/3;
    sigma_Y(i) =  cur_linearity_data(i,3);
    [Xe,Ye] = ellipse(X(i),Y(i),sigma_X(i),sigma_Y(i),32);
    plot(Xe,Ye,'color',[0.5 0.1 0.1])
end

N_plot=2;
X_plot=linspace(min(hou_linearity_data(:,1)), ...
    max(hou_linearity_data(:,1)),N_plot);
Y_plothou=a_hou+b_hou*X_plot;
Y_plotcur=a_cur+b_cur*X_plot;

h(1)=plot(X_plot,Y_plothou,'b','linewidth',2);
h(2)=plot(X_plot,Y_plotcur,'r','linewidth',2);

hold off;
axis equal;
xmin = min(hou_linearity_data(:,1))-10;
xmax = max(hou_linearity_data(:,1))*1.1;
ymin = 0;
ymax = max(max(hou_linearity_data(:,2)),max(cur_linearity_data(:,2)))*1.2;
axis([xmin,xmax,ymin,ymax]);
legend(h,'Hough Transform', 'Curvature Method','Location','northwest')

dim = [0.38 0.55 0.3 0.3];
str = {'Linear Fitting Error', ...
    ['Hough Transform: ' num2str(error_hou)], ...
    ['Curvature Method: ' num2str(error_cur)]};

annotation('textbox',dim,'String',str,'FitBoxToText','on');

xlabel('Manually Counted Bubble Number');
ylabel('Algorithm Counted Bubble Number');
title('Algorithm Performance Comparison');
