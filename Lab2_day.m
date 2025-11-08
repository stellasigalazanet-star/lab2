clear all
close all
clc

station = 16754;
region  = 'europe';

YY      = 2023;
MM      = 11;
DD      = 15;
HH      = 0;

%DATE = strcat( num2str(YY), num2str(MM,'%2.2i'), num2str(DD,'%2.2i'), num2str(HH,'%2.2i') );
filename=['RWS_' num2str(station) '_' datestr(datenum(YY,MM,DD),12) '.htm'];

%% A download file

downloadUA(region,station,YY,MM,DD,HH)

%% B import matrix

[startRow,endRow]=read_UA_html(filename);
%[formatSpec,nvars,startRow,endRow]=read_UA_html(filename);

mat = importfile( filename, startRow, endRow );

PRES    = table2array( mat(:,1) );                 % hPa
HGHT    = table2array( mat(:,2) );                 % m
TEMP    = table2array( mat(:,3) );                 % C
DWPT    = table2array( mat(:,4) );                 % C
RHUM    = table2array( mat(:,5) );                 % g/kg
MIXR    = table2array( mat(:,6) );                 % g/kg
DRCT    = table2array( mat(:,7) );                 % deg
WSPD    = table2array( mat(:,8) ) * 0.514;         % m/s

EVPR    = (29/18) * PRES .* MIXR/1000;             % hPa

%% C plot matrix

figure(1)

plot(PRES,HGHT/1000);grid on;xlabel('PRESSURE (hPa)');ylabel('HEIGHT (km)')
hold on
plot(1013.25*10.^(-HGHT/17000),HGHT/1000,'--');
xlim([0 1050])
legend('sounding','standard atmosphere')
set(gca,'FontSize',14)
gammaP01_02 = (PRES(7)-PRES(4))/(HGHT(7)-HGHT(4));
gammaP09_10 = (PRES(44)-PRES(38))/(HGHT(44)-HGHT(38));


figure(2)

subplot(2,3,1)
plot(TEMP,HGHT/1000);hold on;plot(DWPT,HGHT/1000);grid on
xlabel('TEMP/TDEW (C)');ylabel('HEIGHT (km)')
gammaT00_02 = (TEMP(11)-TEMP(1))/(HGHT(11)-HGHT(1));
gammaT01_11 = (TEMP(52)-TEMP(1))/(HGHT(52)-HGHT(1));

subplot(2,3,2)
plot(MIXR,HGHT/1000);grid on;xlabel('MIXR (g/kg)');ylabel('HEIGHT (km)')
hold on
plot(MIXR(2)*10.^(-HGHT/7000),HGHT/1000,'--');
%plot(MIXR(1)*10.^(-HGHT/5000),HGHT/1000,'--');
legend('sounding','standard atmosphere')
mixrPRC=MIXR/sum(MIXR);
wvapor00_05 = sum(mixrPRC(1:22));

subplot(2,3,3)
plot(EVPR,HGHT/1000);grid on;xlabel('EVPR (hPa)');ylabel('HEIGHT (km)')
hold on
plot(EVPR(2)*10.^(-HGHT/5000),HGHT/1000,'--');
%plot(EVPR(1)*10.^(-HGHT/4000),HGHT/1000,'--');
legend('sounding','standard atmosphere')

subplot(2,3,4)
plot(RHUM,HGHT/1000);grid on;xlabel('RHUM (%)');ylabel('HEIGHT (km)')
xlim([0 100])

subplot(2,3,5)
plot(WSPD,HGHT/1000);grid on;xlabel('WSPD (m/s)');ylabel('HEIGHT (km)')
hold on
%p = polyfit(HGHT,WSPD,4);y1 = polyval(p,HGHT);plot(y1(1:end-1),HGHT(1:end-1)/1000)
[~,jj]=min(abs(HGHT-20000));
p = polyfit(HGHT(1:jj),WSPD(1:jj),4);y1 = polyval(p,HGHT(1:jj));plot(y1(1:end-1),HGHT(1:jj-1)/1000)

subplot(2,3,6)
plot(DRCT,HGHT/1000);grid on;xlabel('WDIR (deg)');ylabel('HEIGHT (km)')
set(gca,'XTick',0:90:360,'XGrid','on');xlim([0 360])

lw = sum(0.5*(MIXR(1:end-1)+MIXR(2:end)).*(-diff(PRES)))/100000;

disp(['Precipitable water = ',num2str(1000*lw,'%10.1f\n'),' mm']) 


