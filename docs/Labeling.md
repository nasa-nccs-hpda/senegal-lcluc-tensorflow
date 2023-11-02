# Image Labeling Methodology

## ArcGIS Pro Image Labeling

### Confirm that Spatial Analyst license is activated

- Open ArcGIS Pro, Go to Settings by clicking the “Project” tab, “Licensing” on the left panel. If it’s not checked at the bottom of the list, then scroll down the panel to “Configure your licensing options”, scroll down the list in the pop up window and check the box next to Spatial Analyst
- Also when using tools, there may be an identical tool from Image Analyst and Spatial Analyst extensions, and to choose the Spatial Analyst Option

### Set Environment

Set “Environments” in the Analysis Tab, choose where you want your work to be saved, and use the drop down to select the original data image or iso image for the other parameters.

- Current Workspace (the default is to use an ArcGIS geodatabase .gdb, but you can create your own folder instead)
- Scratch Workspace
- Output Coordinate System “WGS_1984_UTM_Zone_48N”
- Extent
- Snap Raster
- 	Cell Size
- Cell Size Projection Method: “Preserve Resolution”
- Mask

### Link Project Space

In ArcCatalog, under “Folder Connections”, right click it and select “Connect to a Folder” in order to create a link to your project space or to where you have stored the data and iso images you wish to label.

### Add Imagery

Add data and iso image to map and change the iso image symbology to ‘Unique Values’.

- Compare the data image (or other relevant reference image) to the iso image to determine the land cover of that iso class. 
- In the table of contents, change the color of each unique value according to the desired land cover color scheme, for example green for tree, blue for water, etc. This will be how the values are selected to use in the reclassify tool. 
- Basically you have 60 or so classes and all you are doing is grouping similar looking classes by color so that you get a coherent looking map.  Once this works to your satisfaction you are ready to go to the next step of actually reclassifying.

### Reclassify

Use the Reclassify tool to change the values to the corresponding pixel value. For example if classes 1 through 5 are all tree, you can specify 1-5 to change to 0.

- “Reclassify” to:
- 0 Tree
- 1 Water
- 2 Urban
- 3 Shadow
- 4 Other

Sometimes, the same pixel class can appear in two different land covers (e.g. water in the river as water in the field, urban on the bare earth road as urban on the fallow field, green tree as green crops/other), so you may need to focus on the agricultural classifications and accept some commission errors in the other classes or manually adjust within areas you delineate.

### Adjustment

- “Create Feature Class” (Data Management Tools) for the area that needs adjustment
- In the Edit Toolbar, “Create”, Draw a polygon around the area to be adjusted, Save Edits
- “Polygon to Raster” (converting it to raster)
- “Extract by Mask” (similar to clipping the iso values within your adjustment area)
- “Reclassify” so that the classes within the area are correctly adjusted (e.g. urban to fallow field class) and the outside NoData pixels as -1 so you can reclassify the entire scene in the next step later -> adjustment_raster
- “Raster Calculator” 
- Con (adjustment_raster >=0, adjustment_raster, latest_iso_raster). This will create a new raster which maintains all the adjusted values you made within your adjustment area and then allows you to tell it to use your lastest working iso raster as the rest.

### Export

When you finish, one option is to use “Copy Raster Tool” to save a new copy with the final name and file path that you want to save your training image

### Accuracy Assessment

- Create Accuracy Assessment Points (Spatial Analyst Tools)
- “Compute Confusion Matrix” (Spatial Analyst Tools) and save table with .dbf extension
- “Table to Excel” (Conversion Tools)
- Copy number of pixels in each class from Attribute Table and the Confusion Matrix values into Excel
