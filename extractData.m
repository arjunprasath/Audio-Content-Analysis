maxCol = 0;
nClique = 0;
nSong = 0;
rfileID = fopen('shs_dataset_test.txt','r');
m = matfile('Data.mat','Writable',true);
VectorLength = zeros(1,3500);
songData = zeros(18000,12001);
nline = 0;
for i=1:24100
    tline = fgets(rfileID);
    nline = nline + 1;
    if tline(1) == '#'
        continue
    end
    
    if tline(1) == '%'
        nClique = nClique + 1;
        songData(nSong+1,1) = nClique;
    end
    
    if tline(1) == 'T' && tline(2) == 'R'
        if tline(3) == 'V'                  % V.tar.gz is not working in MSD
            continue
        end
        nSong = nSong + 1;
        disp(['looking into ',num2str(nSong),'th song']);
        
        filename =  [tline(1:18) '.h5'];
        path = fullfile('MSD',tline(3),tline(4),tline(5),filename);
        h5file = h5read(path,'/analysis/segments_pitches');
        if size(h5file,2)>3500
            continue
        end
        if maxCol < size(h5file,2)
            maxCol = size(h5file,2);
        end
        Y = fft2(h5file,12,1000);
        Y = abs(Y);
        for k = 0:11
            for m = 2:1001
                songData(nSong,k*1000+m) = Y(k+1,m-1); 
            end
        end
        songData(nSong,1) = nClique;
        VectorLength(1,size(h5file,2)) = VectorLength(1,size(h5file,2)) +1;

    end
    if tline == -1
        break
    end
    
    %fclose(wfileID);
    
    
    
    
end
save('test_set.mat','songData');
fclose(rfileID);




