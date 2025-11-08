function [startRow,endRow]=read_UA_html(filein)

%first step to find the number of empty rows at the beginning of data block (considered header lines):
fid=fopen(filein);
nheader=0;
endlabel=[];
while 1
    tline = fgetl(fid);
    endlabel=strfind(tline,'----');%Find first line containing '=' to find the number of header lines
    if length(endlabel)>3,break, end
    nheader=nheader+1;
end
fclose(fid);
startRow=nheader+5;

fid=fopen(filein);
nheader=0;
endlabel=[];
while 1
    tline = fgetl(fid);
    endlabel=strfind(tline,'Station information');%Find first line containing '=' to find the number of header lines
    if length(endlabel)>0,break, end
    nheader=nheader+1;
end
fclose(fid);
endRow=nheader-1;
