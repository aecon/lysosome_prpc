/*
 * ImageJ macro that reads all .vsi files inside a directory, 
 * splits the .vsi file into its separate channels and 
 * saves each channel as a separate .tif file (needed for ilastik).
 * 
 * NOTE: Open Plugins -> Bio-Formats -> Bio-Formats Plugins Configurations.
 * 		 Then the Tab Formats, select "CellSens VSI" and check the Windowless box.
 * 		 This loads the .vsi files without opening a popup window with import options.
 */

input_directory = getDirectory("Choose plate directory containing all the .vsi files");

outdir = input_directory.substring(0, lengthOf(input_directory)-1) + "_IJsplit/";
File.makeDirectory(outdir);


files = getFileList(input_directory);
Nfiles = files.length;

setBatchMode(true);

for (i=0; i<Nfiles; i++) {
	if ( endsWith(files[i], ".vsi") ) {
		print(i + "/" + Nfiles);
		//print(files[i]);

		open(input_directory + files[i]);
		t0 = getTitle();

		getDimensions(width, height, channels, slices, frames);
		//print(channels);

		// Get basename
		title = files[i];
		basename = title.substring(0,lengthOf(title)-4);  // currently ends in .vsi
		//print(basename);

		for ( j=1; j<=channels; j++ ){
			selectWindow(t0);
			run("Make Subset...", "channels="+j);
			t1 = getTitle();

			outpath = outdir + basename + "CH"+j+".tif";
			//print("Saving to:", outpath);
			saveAs("Tiff",  outpath);
			close(t1);
		}
		close("*");
	}
}