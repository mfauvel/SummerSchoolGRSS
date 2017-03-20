# Compute the different filters with a template of size 3x3 and 11x11
for i in 3 11
do
    # Mean filter
    otbcli_BandMathX -il ../Data/pca_university.tif -out ../Data/pca_mean_${i}_${i}_university.tif \
		     -exp "mean(im1b1N${i}x${i}); mean(im1b2N${i}x${i}); mean(im1b3N${i}x${i})"

    # Var filter
    otbcli_BandMathX -il ../Data/pca_university.tif -out ../Data/pca_std_${i}_${i}_university.tif \
		     -exp "var(im1b1N${i}x${i}); var(im1b2N${i}x${i}); var(im1b3N${i}x${i})"

    # Range filter
    otbcli_BandMathX -il ../Data/pca_university.tif -out ../Data/pca_range_${i}_${i}_university.tif \
		     -exp "vmax(im1b1N${i}x${i})-vmin(im1b1N${i}x${i}); vmax(im1b2N${i}x${i})-vmin(im1b2N${i}x${i});\
                     vmax(im1b3N${i}x${i})-vmin(im1b3N${i}x${i})"

    # Median filter
    otbcli_BandMathX -il ../Data/pca_university.tif -out ../Data/pca_median_${i}_${i}_university.tif \
		     -exp "median(im1b1N${i}x${i}); median(im1b2N${i}x${i}); median(im1b3N${i}x${i})"
done
