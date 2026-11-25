// TROPOMI OFFL L3 NO2: 3x3 native-grid windows around the 24 Madrid stations for the 236 panel dates.
// Purpose: raster-neighbour evidence for the three-way relaxation study (probabilistic / possibilistic / evidential).
// UNTESTED draft: first run with TEST=true (5 dates), check the Tasks output, then set TEST=false.
var TEST = true;
var dates = ["2025-01-01", "2025-01-02", "2025-01-10", "2025-01-12", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17", "2025-01-18", "2025-01-19", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-08", "2025-02-09", "2025-02-12", "2025-02-13", "2025-02-14", "2025-02-16", "2025-02-17", "2025-02-19", "2025-02-20", "2025-02-23", "2025-02-26", "2025-02-27", "2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31", "2025-04-01", "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-14", "2025-04-17", "2025-04-20", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-26", "2025-04-27", "2025-05-01", "2025-05-08", "2025-05-11", "2025-05-14", "2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-24", "2025-05-25", "2025-05-26", "2025-05-27", "2025-05-28", "2025-05-29", "2025-05-30", "2025-05-31", "2025-06-01", "2025-06-02", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15", "2025-06-16", "2025-06-17", "2025-06-18", "2025-06-20", "2025-06-21", "2025-06-22", "2025-06-24", "2025-06-25", "2025-06-26", "2025-06-27", "2025-06-28", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05", "2025-07-06", "2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11", "2025-07-12", "2025-07-13", "2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-19", "2025-07-20", "2025-07-21", "2025-07-22", "2025-07-24", "2025-07-25", "2025-07-26", "2025-07-27", "2025-07-28", "2025-07-29", "2025-07-30", "2025-07-31", "2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05", "2025-08-07", "2025-08-08", "2025-08-09", "2025-08-10", "2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14", "2025-08-15", "2025-08-16", "2025-08-17", "2025-08-19", "2025-08-20", "2025-08-21", "2025-08-22", "2025-08-23", "2025-08-24", "2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30", "2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05", "2025-09-06", "2025-09-08", "2025-09-09", "2025-09-11", "2025-09-12", "2025-09-13", "2025-09-14", "2025-09-15", "2025-09-16", "2025-09-17", "2025-09-20", "2025-09-21", "2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-26", "2025-09-29", "2025-10-01", "2025-10-02", "2025-10-04", "2025-10-05", "2025-10-07", "2025-10-09", "2025-10-12", "2025-10-13", "2025-10-14", "2025-10-15", "2025-10-16", "2025-10-18", "2025-10-23", "2025-10-28", "2025-11-04", "2025-11-06", "2025-11-08", "2025-11-09", "2025-11-10", "2025-11-12", "2025-11-18", "2025-11-19", "2025-11-25", "2025-11-26", "2025-11-27", "2025-11-28", "2025-11-30", "2025-12-01", "2025-12-03", "2025-12-13", "2025-12-17", "2025-12-18", "2025-12-23", "2025-12-24", "2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-09", "2026-01-11", "2026-01-19", "2026-01-20", "2026-01-31", "2026-02-08", "2026-02-12", "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-02-28"];
if (TEST) dates = dates.slice(70, 75);
var stations = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-3.7122567, 40.4238823]), {code: 4, name: 'Plaza de España'}),
  ee.Feature(ee.Geometry.Point([-3.6823158, 40.4215533]), {code: 8, name: 'Escuelas Aguirre'}),
  ee.Feature(ee.Geometry.Point([-3.6773491, 40.4514734]), {code: 11, name: 'Ramón y Cajal'}),
  ee.Feature(ee.Geometry.Point([-3.6392422, 40.4400457]), {code: 16, name: 'Arturo Soria'}),
  ee.Feature(ee.Geometry.Point([-3.7133167, 40.347147]), {code: 17, name: 'Villaverde'}),
  ee.Feature(ee.Geometry.Point([-3.7318356, 40.3947825]), {code: 18, name: 'Farolillo'}),
  ee.Feature(ee.Geometry.Point([-3.7473445, 40.4193577]), {code: 24, name: 'Casa de Campo'}),
  ee.Feature(ee.Geometry.Point([-3.5800258, 40.4769179]), {code: 27, name: 'Barajas Pueblo'}),
  ee.Feature(ee.Geometry.Point([-3.7031662, 40.4192091]), {code: 35, name: 'Plaza del Carmen'}),
  ee.Feature(ee.Geometry.Point([-3.6453104, 40.4079517]), {code: 36, name: 'Moratalaz'}),
  ee.Feature(ee.Geometry.Point([-3.7071303, 40.4455439]), {code: 38, name: 'Cuatro Caminos'}),
  ee.Feature(ee.Geometry.Point([-3.7115364, 40.4782322]), {code: 39, name: 'Barrio del Pilar'}),
  ee.Feature(ee.Geometry.Point([-3.6515286, 40.3881478]), {code: 40, name: 'Vallecas'}),
  ee.Feature(ee.Geometry.Point([-3.6868138, 40.3980991]), {code: 47, name: 'Méndez Álvaro'}),
  ee.Feature(ee.Geometry.Point([-3.6903729, 40.4398904]), {code: 48, name: 'Castellana'}),
  ee.Feature(ee.Geometry.Point([-3.6824999, 40.4144444]), {code: 49, name: 'Parque del Retiro'}),
  ee.Feature(ee.Geometry.Point([-3.6887449, 40.4655841]), {code: 50, name: 'Plaza Castilla'}),
  ee.Feature(ee.Geometry.Point([-3.6121394, 40.3730118]), {code: 54, name: 'Ensanche de Vallecas'}),
  ee.Feature(ee.Geometry.Point([-3.5805649, 40.4623628]), {code: 55, name: 'Urb. Embajada'}),
  ee.Feature(ee.Geometry.Point([-3.7187679, 40.3850336]), {code: 56, name: 'Plaza Elíptica'}),
  ee.Feature(ee.Geometry.Point([-3.6605173, 40.4942012]), {code: 57, name: 'Sanchinarro'}),
  ee.Feature(ee.Geometry.Point([-3.7746101, 40.5180701]), {code: 58, name: 'El Pardo'}),
  ee.Feature(ee.Geometry.Point([-3.609031, 40.465144]), {code: 59, name: 'Juan Carlos I'}),
  ee.Feature(ee.Geometry.Point([-3.6897308, 40.5005477]), {code: 60, name: 'Tres Olivos'})
]);
var BAND = 'tropospheric_NO2_column_number_density';          // mol/m2, OFFL product (QA-filtered by the catalogue)
var S5P = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').select(BAND);
var proj = S5P.first().projection();                          // native grid, fixed for every date
var scale = proj.nominalScale();

function perDate(d) {
  d = ee.Date(d);
  var c = S5P.filterDate(d, d.advance(1, 'day'));
  var n = c.size();
  var fc = ee.Algorithms.If(n.gt(0), (function() {
    // documented rule: pixel-wise mean of all valid same-day observations (no order-dependent mosaic)
    var img = c.mean().rename('v').reproject(proj);
    var cnt = c.count().rename('n').reproject(proj);
    var win = img.neighborhoodToBands(ee.Kernel.square({radius: 1, units: 'pixels'}));   // bands v_-1_-1 ... v_1_1
    var all = win.addBands(cnt);
    return all.reduceRegions({collection: stations, reducer: ee.Reducer.first(), scale: scale, crs: proj})
              .map(function(f) { return f.set({date: d.format('YYYY-MM-dd'), n_images: n}); });
  })(), ee.FeatureCollection([]));
  return ee.FeatureCollection(fc);
}
var out = ee.FeatureCollection(ee.List(dates).map(perDate)).flatten();
Export.table.toDrive({
  collection: out, description: 'tropomi_station_windows' + (TEST ? '_TEST' : ''), fileFormat: 'CSV',
  selectors: ['date','code','name','n_images','n',
    'v_-1_-1','v_0_-1','v_1_-1','v_-1_0','v_0_0','v_1_0','v_-1_1','v_0_1','v_1_1']
});
// scene metadata: whatever properties the catalogue provides (processor/algorithm versions are NOT assumed by name)
var meta = S5P.filterDate(dates[0], ee.Date(dates[dates.length-1]).advance(1,'day'));
Export.table.toDrive({
  collection: ee.FeatureCollection(meta.map(function(img) {
    return ee.Feature(null, img.toDictionary()).set({id: img.id(), time_start: img.get('system:time_start')}); })),
  description: 'tropomi_scene_metadata' + (TEST ? '_TEST' : ''), fileFormat: 'CSV'
});
print('projection', proj, 'nominal scale (m)', scale, 'dates', dates.length);
