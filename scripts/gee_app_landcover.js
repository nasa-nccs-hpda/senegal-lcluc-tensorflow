/*
  The following application is part of the "" paper.
  The outputs from this GEE App provide blabla.
  Classes from the map:
    1: Other ()
    2: Tree ()
    3: Tree/Other Tie ()
    4: Crop ()
    5: Crop/Other Tie ()
    6: Tree/Crop Tie ()
    7: Tree/Crop/Other Tie ()
    10: No-data ()
*/

// ===========================================================
// Data Definition Section
// ===========================================================

// Senegal ROI
var senegalBoundary = ee.FeatureCollection(
  'projects/ee-senegal-lcluc/assets/ancillary/Senegal_Region')
var senegalCASBoundary = ee.FeatureCollection(
  'projects/ee-senegal-lcluc/assets/ancillary/Senegal_CAS_Region')
var senegalETZBoundary = ee.FeatureCollection(
  'projects/ee-senegal-lcluc/assets/ancillary/Senegal_ETZ_Region')

// Internal Data Products
var casComposite = ee.Image('projects/ee-senegal-lcluc/assets/composites/CAS_composite_v4-2_2009-2023_modepost-unclipped');
var etzComposite = ee.Image('projects/ee-senegal-lcluc/assets/composites/ETZ_composite_v4-2_2009-2023_modepost-unclipped');

// External Data Products
var gladCropland = ee.ImageCollection('users/potapovpeter/Global_cropland_2019');
var worldCov10m = ee.ImageCollection('ESA/WorldCover/v100').first();

// Cropping to what?
worldCov10m = worldCov10m.clip(senegalBoundary)

// ===========================================================
// Preprocessing the Data
// ===========================================================

// ----------------------- Senegal LCLUC ---------------------

// Define a palette for the 18 distinct land cover classes.
// 1, 7: #785f25
// 2, 3: #33a02c
// 4, 5, 6: #ff7f00
// 10: transparent
var senegalPalette = [
  '#785f25', // Other
  '#33a02c', // Tree
  '#33a02c', // Tree/Other Tie
  '#ff7f00', // Crop
  '#ff7f00', // Crop/Other Tie
  '#ff7f00', // Tree/Crop Tie
  '#785f25', // Tree/Crop/Other Tie
];

casComposite = casComposite.clip(senegalCASBoundary)
etzComposite = etzComposite.clip(senegalETZBoundary)

casComposite = casComposite.updateMask(casComposite.lt(8));
casComposite = casComposite.updateMask(casComposite.gt(0));
var casCompositeLayer = ui.Map.Layer(
  casComposite, {palette:senegalPalette, min:1, max:7}, 'Senegal LCLUC CAS', true);
Map.add(casCompositeLayer);

etzComposite = etzComposite.updateMask(etzComposite.lt(8));
etzComposite = etzComposite.updateMask(etzComposite.gt(0));
var etzCompositeLayer = ui.Map.Layer(
  etzComposite, {palette:senegalPalette, min:1, max:7}, 'Senegal LCLUC ETZ', true);
Map.add(etzCompositeLayer);


// ----------------------- GLAD Cropland ---------------------

// glad cropland clip
gladCropland = gladCropland.map(function(image){
  return image.clip(senegalBoundary)
})

// mask cropland
var gladCroplandMasked = gladCropland.map(function(image){
  var masked = image.gt(0)
  return image.updateMask(masked)
})

var gladCroplandLayer = ui.Map.Layer(
  gladCroplandMasked, {palette:['#FFA500'], min:0, max:1}, 'Glad Cropland', false)

Map.add(gladCroplandLayer);

// Add WorldCov
var visualizationWorldCov10m = {
  bands: ['Map'],
};
var worldCov10mLayer = ui.Map.Layer(worldCov10m, visualizationWorldCov10m, 'WorldCov 10m', false)
Map.add(worldCov10mLayer);

// ===========================================================
// Appearance Options
// ===========================================================

// Set up a satellite background
Map.setOptions('Satellite')

// Center the map to Guyana
Map.centerObject(senegalBoundary, 8)

// ----------------------- Legend ---------------------

var senegalPaletteShort = [
  '785f25', // Other
  '33a02c', // Tree
  'ff7f00', // Crop
];

var senegalPaletteNames = [
  'Other', // Other
  'Tree', // Tree
  'Cropland', // Crop
];

// set position of panel
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});
 
// Create legend title
var legendTitle = ui.Label({
  value: 'LCLUC Legend',
  style: {
    fontWeight: 'bold',
    fontSize: '18px',
    margin: '0 0 4px 0',
    padding: '0'
    }
});
 
// Add the title to the panel
legend.add(legendTitle);
 
// Creates and styles 1 row of the legend.
var makeRow = function(color, name) {
 
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: '#' + color,
          // Use padding to give the box height and width.
          padding: '8px',
          margin: '0 0 4px 0'
        }
      });
 
      // Create the label filled with the description text.
      var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });
 
      // return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};
 
// Add color and and names
for (var i = 0; i < 3; i++) {
  legend.add(makeRow(senegalPaletteShort[i], senegalPaletteNames[i]));
  }  
 
// add legend to map (alternatively you can also print the legend to the console)
Map.add(legend);

///////////////////////////////////////////////////////////////
//      3) Set up panels and widgets for display             //
///////////////////////////////////////////////////////////////

//3.1) Set up title and summary widgets

//App title
var header = ui.Label('Navigating deep learning strategies for large-area land cover mapping using very-high-resolution imagery in Senegal', {fontSize: '20px', fontWeight: 'bold'});

// GLAD Extent Checkbox
var extCheckGLAD = ui.Checkbox('Glad Cropland').setValue(false);
var doCheckboxGLAD = function() {
  extCheckGLAD.onChange(function(checked){
  gladCroplandLayer.setShown(checked)
  })
}
doCheckboxGLAD();

// VHR Extent Checkbox
var extCheckVHR = ui.Checkbox('VHR LCLUC').setValue(false);
var doCheckboxVHR = function() {
  extCheckVHR.onChange(function(checked){
  etzCompositeLayer.setShown(checked)
  casCompositeLayer.setShown(checked)
  })
}
doCheckboxVHR();

// GLAD Extent Checkbox
var extCheckWCov = ui.Checkbox('WorldCov 2021').setValue(false);
var doCheckboxWCov = function() {
  extCheckWCov.onChange(function(checked){
  worldCov10mLayer.setShown(checked)
  })
}
doCheckboxWCov();

var text1 = ui.Label(
  'Rapid advances in deep learning for land cover classification of trees, shrubs and very small agricultural fields using very high-resolution satellite data (< 2 m), has tremendous potential for resolving current challenges in quantifying land cover change in sub-Saharan African (SSA), due to growing demand for food resources. We conducted experiments with different training strategies for scaling up UNet convolutional neural network models for regional land cover mapping with multispectral WorldView (WV)-2 and –3, imagery in three distinct regions of Senegal which has complex seasonal wet/dry conditions and cropland-savanna mosaics.',
    {fontSize: '15px'});

var text2 = ui.Label(
  'Paper Information:',
    {fontSize: '15px', fontWeight: 'bold'});

//App summary
var text3 = ui.Label(
  'Wessels, K., Le, M. T., Caraballo-Vega, J. A., Wooten, M., Carroll, M., Brown, M. E., Aziz Diouf, A., Mbaye, M., Neigh, C. (2024). Navigating deep learning strategies for large-area land cover mapping using very-high-resolution imagery in Senegal. Submitted to International Journal of Applied Earth Observation and Geoinformation.',
    {fontSize: '15px'});

var text4 = ui.Label(
  'Citing the Dataset:',
    {fontSize: '15px', fontWeight: 'bold'});

var text5 = ui.Label(
  'Wessels, K., Le, M. T., Caraballo-Vega, J. A., Wooten, M., Carroll, M., Brown, M. E., Aziz Diouf, A., Mbaye, M., & Neigh, C. (2024). Navigating deep learning strategies for large-area land cover mapping using very-high-resolution imagery in Senegal: Composite Outputs [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13975528.',
    {fontSize: '15px'});
    

//3.2) Create a panel to hold text
var panel = ui.Panel({
  widgets:[header],//Adds header and text
  style:{width: '300px',position:'middle-right'}});

var extLabel = ui.Label(
  'Land Cover Products:',
    {fontSize: '15px', fontWeight: 'bold'});

var extLabelWorldCovLegend = ui.Label({value:'World Cov Legend',
style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
});

var extLabelGladLegend = ui.Label({value:'GLAD Cropland Legend',
style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
});

var extLabelVHR = ui.Label({value:'VHR LCLUC Legend',
style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
});

// Set position of panel
var extentLegendWCov = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var extentLegendGlad = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var extentLegendVHR = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});



// The following creates and styles 1 row of the legend.
var makeRowa = function(color, name) {
 
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: '#' + color,
          // Use padding to give the box height and width.
          padding: '8px',
          margin: '0 0 4px 0'
        }
      });
 
      // Create a label with the description text.
      var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });
 
      // Return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};


//Create a palette using the same colors we used for each extent layer
var paletteMAPa = [
  "006400",
  "ffbb22",
  "ffff4c",
  "f096ff",
  "fa0000",
  "b4b4b4",
  "f0f0f0",
  "0064c8",
  "0096a0",
  "00cf75",
  "fae6a0",
];


// Name of each legend value
var namesa = [
  "10 Trees", "20 Shrubland", "30 Grassland", 
  "40 Cropland", "50 Built-up", "60 Barren / sparse vegetation",
  "70 Snow and ice", "80 Open water", "90 Herbaceous wetland",
  "95 Mangroves", "100 Moss and lichen"
]; 
           
 
// Add color and names to legend
for (var i = 0; i < paletteMAPa.length; i++) {
  extentLegendWCov.add(makeRowa(paletteMAPa[i], namesa[i]));
  }  

var paletteMAPVHR = [
  '785f25', // Other
  '33a02c', // Tree
  'ff7f00', // Crop
];
var namesVHR = [
  "Other", "Trees", "Cropland"
]; 

// Add color and names to legend
for (var i = 0; i < paletteMAPVHR.length; i++) {
  extentLegendVHR.add(makeRowa(paletteMAPVHR[i], namesVHR[i]));
  } 
  
extentLegendGlad.add(makeRowa('ffa500', 'Cropland'));

//4.4) Add these new widgets to the panel in the order you want them to appear
panel.add(extLabel)
      .add(extCheckVHR)
      .add(extCheckGLAD)
      .add(extCheckWCov)
      .add(extLabelVHR)
      .add(extentLegendVHR)
      .add(extLabelWorldCovLegend)
      .add(extentLegendWCov)
      .add(extLabelGladLegend)
      .add(extentLegendGlad)
      .add(text1)
      .add(text2)  
      .add(text3)
      .add(text4) 
      .add(text5) 

//3.3) Create variable for additional text and separators

//This creates another panel to house a line separator and instructions for the user
var intro = ui.Panel([
  ui.Label({
    value: '____________________________________________',
    style: {fontWeight: 'bold'},
  })
]);

//Add this new panel to the larger panel we created 
panel.add(intro)

//3.4) Add our main panel to the root of our GUI
ui.root.insert(1,panel)

