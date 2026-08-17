// Rebuild the public Sentinel-1/2 temporal exports used by this repository.
//
// IMPORTANT: the historical GeoTIFF/CSV were deleted before this script was
// recovered. This reconstruction preserves the documented 23 anchors, band
// schema, public collections, Shanghai grid and frozen sample IDs, but it must
// be treated as a new data version unless its SHA-256 matches data_registry.yaml.
// Upload data_metadata/alphaearth_sample_manifest.csv as a table and replace
// MANIFEST_ASSET below. Only rows from the Jiangxi source are sampled.

var MANIFEST_ASSET = 'projects/REPLACE_ME/assets/alphaearth_sample_manifest';
var anchors = [
  '2022-03-02', '2022-03-12', '2022-04-06', '2022-04-11', '2022-04-21',
  '2022-05-06', '2022-05-16', '2022-06-25', '2022-07-10', '2022-08-04',
  '2022-08-09', '2022-08-19', '2022-08-24', '2022-09-13', '2022-09-18',
  '2022-09-28', '2022-10-03', '2022-10-13', '2022-10-18', '2022-10-23',
  '2022-11-02', '2022-11-07', '2022-11-12'
];

function suffix(i) { return ee.Number(i).add(1).format('%02d'); }

function s2At(date, i, region) {
  date = ee.Date(date);
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region).filterDate(date.advance(-8, 'day'), date.advance(9, 'day'))
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 80));
  var ndvi = collection.map(function(image) {
    var scl = image.select('SCL');
    var clear = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
    return image.updateMask(clear).normalizedDifference(['B8', 'B4']);
  }).median();
  return ndvi.rename(ee.String('s2_ndvi_t').cat(suffix(i)));
}

function s1At(date, i, region) {
  date = ee.Date(date);
  var image = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region).filterDate(date.advance(-8, 'day'), date.advance(9, 'day'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']).median();
  var vv = image.select('VV');
  var vh = image.select('VH');
  var vvLinear = ee.Image(10).pow(vv.divide(10));
  var vhLinear = ee.Image(10).pow(vh.divide(10));
  var rvi = vhLinear.multiply(4).divide(vvLinear.add(vhLinear));
  var tag = suffix(i);
  return vv.rename(ee.String('s1_vv_t').cat(tag))
    .addBands(vh.rename(ee.String('s1_vh_t').cat(tag)))
    .addBands(rvi.rename(ee.String('s1_rvi_t').cat(tag)));
}

function buildStack(region) {
  var stack = ee.Image([]);
  anchors.forEach(function(date, i) { stack = stack.addBands(s2At(date, i, region)); });
  anchors.forEach(function(date, i) { stack = stack.addBands(s1At(date, i, region)); });
  return stack;
}

var manifest = ee.FeatureCollection(MANIFEST_ASSET);
var source = manifest.filter(ee.Filter.eq('region', 'jiangxi'));
var shanghaiBounds = ee.Geometry.Rectangle(
  [362320, 3488900, 391060, 3510320], 'EPSG:32651', false
);
var sourceStack = buildStack(source.geometry().bounds());
var targetStack = buildStack(shanghaiBounds);

var sampled = sourceStack.sampleRegions({
  collection: source,
  properties: ['sample_id', 'class_id', 'label_source', 'spatial_block', 'split'],
  scale: 20,
  geometries: true,
  tileScale: 4
});

Export.table.toDrive({
  collection: sampled,
  description: 'jiangxi_official_2022_binary_features_gee_s1_s2_rebuilt',
  fileNamePrefix: 'jiangxi_official_2022_binary_features_gee_s1_s2',
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: targetStack,
  description: 'shanghai_light_2022_fusion_92_20m_rebuilt',
  fileNamePrefix: 'shanghai_light_2022_fusion_92_20m',
  region: shanghaiBounds,
  crs: 'EPSG:32651',
  crsTransform: [20, 0, 362320, 0, -20, 3510320],
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: -9999}
});

print('source rows requested', source.size());
print('source rows exported after masks', sampled.size());
print('source feature bands', sourceStack.bandNames());
