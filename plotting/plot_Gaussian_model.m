set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[35,15]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,35,15]) % Place plot on figure

thresholds = 0:0.1:1;
%subplot(1,3,1);
hold on;
plot(thresholds, gUnseenAccuracies, 'r-+', 'LineWidth', 2);
plot(thresholds, gSeenAccuracies, 'r-+', 'LineWidth', 2);
h_title = title('Gaussian model');
h_xl = xlabel('Fraction of points classified as unseen');
h_yl = ylabel('Accuracy');
set(h_title, 'FontSize', 24);
set(h_xl, 'FontSize', 24);
set(h_yl, 'FontSize', 24);

file_name = './figures/gaussian_model_acc.jpg';
Image = getframe(gcf);
imwrite(Image.cdata, file_name);