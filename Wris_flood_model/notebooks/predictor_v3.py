import logging, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from app.models.schemas import CollectedFeatures
from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    risk_level:          str
    confidence:          float
    risk_score:          float
    class_probabilities: dict

class FloodPredictor:
    def __init__(self):
        self._model=self._le=self._num_imp=self._scaler=None
        self._num_cols=self._cat_imp=self._cat_cols=self._feat_names=self._clip_bounds=None
        self.is_loaded=False; self.model_name='not loaded'

    def load_model(self, path=None):
        model_path = Path(path or settings.MODEL_PATH)
        if not model_path.exists(): raise FileNotFoundError(f'Model not found at {model_path}')
        with open(model_path, 'rb') as f: saved = pickle.load(f)
        self._model=saved['model']; self._le=saved['label_encoder']
        self._feat_names=saved['feature_names']; self._num_imp=saved['num_imputer']
        self._scaler=saved['scaler']; self._num_cols=saved['num_cols']
        self._cat_imp=saved['cat_imputer']; self._cat_cols=saved['cat_cols']
        self._clip_bounds=saved['clip_bounds']
        self.model_name=type(self._model).__name__; self.is_loaded=True
        logger.info(f'Model: {self.model_name} | features: {len(self._feat_names)} | classes: {list(self._le.classes_)}')

    def predict(self, lat, lon, date, state, district, flood_zone, features):
        if not self.is_loaded: raise RuntimeError('Model not loaded.')
        X=self._build_feature_vector(lat,lon,date,state,district,flood_zone,features)
        proba=self._model.predict_proba(X)[0]; pred_idx=int(np.argmax(proba))
        classes=list(self._le.classes_)
        class_probs={cls:round(float(p),4) for cls,p in zip(classes,proba)}
        risk_level=self._le.inverse_transform([pred_idx])[0]
        confidence=round(float(proba[pred_idx]),4)
        weights={'Low':0.0,'Moderate':0.5,'High':1.0}
        risk_score=round(sum(weights.get(c,0.5)*p for c,p in class_probs.items()),4)
        logger.info(f'Prediction: {risk_level} ({confidence:.1%}) | {class_probs}')
        return PredictionResult(risk_level=risk_level,confidence=confidence,
                                risk_score=risk_score,class_probabilities=class_probs)

    def _build_feature_vector(self,lat,lon,date,state,district,flood_zone,features):
        month=int(date[5:7])
        raw={'rainfall_mm':float(features.rainfall_mm or 0.0),
             'temperature_c':float(features.temperature_c or 25.0),
             'humidity_pct':float(features.humidity_pct or 65.0),
             'river_discharge_m3_s':float(features.river_discharge_m3_s or 50.0),
             'water_level_m':float(features.water_level_m or 100.0),
             'soil_moisture':float(features.soil_moisture or 15.0),
             'atmospheric_pressure':float(features.atmospheric_pressure or 1010.0),
             'evapotranspiration':float(features.evapotranspiration or 2.0)}
        for col,(lo,hi) in self._clip_bounds.items(): raw[col]=float(np.clip(raw[col],lo,hi))
        raw['rainfall_mm']=np.log1p(max(raw['rainfall_mm'],0))
        raw['river_discharge_m3_s']=np.log1p(max(raw['river_discharge_m3_s'],0))
        raw['water_level_m']=np.log1p(max(raw['water_level_m'],0))
        hum=np.clip(raw['humidity_pct'],0,100)
        raw['rain_x_humidity']=raw['rainfall_mm']*hum/100.0
        raw['discharge_x_wlevel']=raw['river_discharge_m3_s']*raw['water_level_m']
        raw['moisture_deficit']=max(raw['evapotranspiration'],0)-max(raw['soil_moisture'],0)
        raw['heat_index']=raw['temperature_c']*hum/100.0
        raw['month_sin']=float(np.sin(2*np.pi*month/12))
        raw['month_cos']=float(np.cos(2*np.pi*month/12))
        raw['is_monsoon']=float(month in [6,7,8,9])
        raw['dist_from_center']=float(np.sqrt((lat-20.5)**2+(lon-78.9)**2))
        raw['latitude']=lat; raw['longitude']=lon
        num_values=np.array([[raw.get(col,0.0) for col in self._num_cols]])
        num_scaled=self._scaler.transform(self._num_imp.transform(num_values))
        num_df=pd.DataFrame(num_scaled,columns=self._num_cols)
        cat_input=pd.DataFrame([{'stateName':state.strip().title(),
            'districtName':district.strip().title(),'flood_zone':flood_zone.strip()}])[self._cat_cols]
        cat_imputed=pd.DataFrame(self._cat_imp.transform(cat_input),columns=self._cat_cols)
        cat_ohe=pd.get_dummies(cat_imputed,drop_first=True)
        row_df=pd.concat([num_df,cat_ohe],axis=1)
        row_df=row_df.reindex(columns=self._feat_names,fill_value=0.0)
        return row_df.values.astype(np.float32)