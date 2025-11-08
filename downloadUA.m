function downloadUA(region,station,YY,MM,DD,HH)

%years=[2021:2021];
%months=[11:11];
%station='16622'
%station='15420'
%region='europe'

%%%%%%%%%%%%%%%%%%%%%

%for Y=years
 %   for M=months
        while 1
            %eom= num2str(eomday(Y, M));
            url=['http://weather.uwyo.edu/cgi-bin/sounding?region=' region '&TYPE=TEXT%3ALIST&YEAR=' num2str(YY) '&MONTH=' num2str(MM,'%2.2i') '&FROM=' strcat(num2str(DD,'%2.2i'),num2str(HH,'%2.2i')) '&TO=' strcat(num2str(DD,'%2.2i'),num2str(HH,'%2.2i')) '&STNM=' num2str(station)]
            filename=['RWS_' num2str(station) '_' datestr(datenum(YY,MM,DD),12) '.htm'];
            urlwrite(url,filename);
            % Control of file dimension!
            DR=dir(filename);
            if DR.bytes>500
                disp('Ok...')
                break%Continue only if dimension exceed 550 bytes. In case not, restart the same request
            else
                disp('Server too busy! Try again in 100 s')
                pause(100)
            end
        end
%     pause
%    end
%end


