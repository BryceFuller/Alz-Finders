
close all;
%clear all;
clc;


for ii = 1:2
    
    file_path = strcat("C:\Users\Reid\Desktop\dataSciPrinciples\finalProj\Alz-Finders\matlab_output_patch_", num2str(ii));
    file_path = strcat(file_path, ".mat"); 

    patch_file = open(file_path);
    patches = patch_file.PatchAD;
    for jj = 1:size(PatchAD,1)
        patch = PatchAD(jj,:,:,20);
        thisImage = reshape(patch,[50, 41]); 
        figure;
        
        imagesc(thisImage);
        colorbar;
        
    end


end



