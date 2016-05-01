fid = fopen('deepLearningParTune.dat', 'r');
fidout = fopen('data.dat', 'w');
tline = fgets(fid);
while ischar(tline)
    if (tline(1:5) == 'batch')
        fprintf(fidout,'\n');
    end
    disp(tline);
    tline = tline(1:end-1);
    fprintf(fidout,'%s',tline);
    fprintf(fidout,'%s',' ');
    tline = fgets(fid);
end
fclose(fidout);
fclose(fid);