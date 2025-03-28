setAutoThreshold("Default dark no-reset");
setThreshold(0.5000, 1000000000000000000000000000000.0000);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Analyze Particles...", "size=500-Infinity show=Masks add");
run("Invert LUT");