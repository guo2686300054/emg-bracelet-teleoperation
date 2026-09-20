import csv,hashlib,json,tempfile,unittest
from pathlib import Path
from legacy_dataset import CHANNEL_COLUMNS,LegacyDatasetError,import_legacy_dataset,import_legacy_measurement_manifest,load_experimental_analysis_groups,load_training_groups,validate_legacy_transform

class LegacyDatasetTests(unittest.TestCase):
 def fixture(self,root,rows,*,factor=1,segments=None,version="1.1",status="completed",session_id="session-s1"):
  root=Path(root);source=root/"client_data.csv"
  with source.open("w",encoding="utf-8",newline="") as stream:w=csv.writer(stream,lineterminator="\n");w.writerow(CHANNEL_COLUMNS);w.writerows(rows)
  if segments is None:segments=[{"start_row":0,"end_row_exclusive":len(rows),"action_label":"rest","action_phase":"rest","confidence":"reported","training_usable_raw":False,"eligible_for_future_training_review":True,"basis":"reported rest"}]
  p={"schema":"emg_session_annotations","version":version,"source_file":"client_data.csv","source_directory_name":root.name,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"source_size_bytes":source.stat().st_size,"source_row_count":len(rows),"session_id":session_id,"status":status,"training_usable_raw":False,"split_group_id":"subject-1","full_session_group_id":f"subject-1:{session_id}","legacy_write_factor":factor,"segments":segments}
  a=root/"annotations.json";a.write_text(json.dumps(p),encoding="utf-8");return source,a
 def batch(self,root,*,first_rows=None,first_segments=None,statuses=None):
  root=Path(root);sessions=[];statuses=statuses or {}
  for factor in range(1,8):
   d=root/f"source-{factor}";d.mkdir();rows=first_rows if factor==1 and first_rows is not None else [self.vec(factor)]*factor
   _,a=self.fixture(d,rows,factor=factor,segments=first_segments if factor==1 else None,status=statuses.get(factor,"completed"),session_id=f"session-s{factor}");p=json.loads(a.read_text())
   sessions.append({"relative_directory":d.name,"annotation_file":"annotations.json","session_id":p["session_id"],"status":p["status"],"training_usable_raw":False,"source_sha256":p["source_sha256"],"source_size_bytes":p["source_size_bytes"],"source_row_count":p["source_row_count"],"legacy_write_factor":factor,"full_session_group_id":p["full_session_group_id"]})
  m=root/"measurement.json";m.write_text(json.dumps({"schema":"emg_measurement_manifest","version":"1.1","session_count":7,"split_group_id":"subject-1","raw_training_policy":{"training_usable_raw":False},"sessions":sessions}),encoding="utf-8")
  out=root/"batch";import_legacy_measurement_manifest(m,out);return m,out
 @staticmethod
 def vec(v):return [v]*8
 @staticmethod
 def rows(result):
  with result.derived_csv.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))

 def test_k1_preserves_legitimate_equal_rows_and_source(self):
  with tempfile.TemporaryDirectory() as root:
   source,a=self.fixture(root,[self.vec(7)]*3);before=source.read_bytes();r=import_legacy_dataset(a,Path(root)/"out");self.assertEqual(r.output_rows,3);self.assertEqual(source.read_bytes(),before)
 def test_factor_chunks_and_incomplete_run_use_ceiling(self):
  with tempfile.TemporaryDirectory() as root:
   _,a=self.fixture(root,[self.vec(9)]*5,factor=2);r=import_legacy_dataset(a,Path(root)/"out");m=json.loads(r.transform_manifest.read_text());self.assertEqual(r.output_rows,3);self.assertEqual(m["raw_to_canonical_ranges"][-1],{"raw_start_row":4,"raw_end_row_exclusive":5,"canonical_index":2});self.assertIsNone(m["signal_metadata"]["sampling_rate_hz"])
   self.assertEqual(validate_legacy_transform(r.transform_manifest).manifest["training_provenance"]["kind"],"legacy_experimental")
 def test_factor_required_schema_exact_and_raw_claim_rejected(self):
  for mutation,pattern in ((lambda p:p.pop("legacy_write_factor"),"legacy_write_factor"),(lambda p:p.update(version="1.2"),"unsupported"),(lambda p:p.update(training_usable_raw=True),"training_usable_raw")):
   with self.subTest(pattern=pattern),tempfile.TemporaryDirectory() as root:
    _,a=self.fixture(root,[self.vec(1)]);p=json.loads(a.read_text());mutation(p);a.write_text(json.dumps(p));
    with self.assertRaisesRegex(LegacyDatasetError,pattern):import_legacy_dataset(a,Path(root)/"out")
 def test_hash_segment_gap_and_bounds_rejected(self):
  with tempfile.TemporaryDirectory() as root:
   source,a=self.fixture(root,[self.vec(1)]);source.write_bytes(source.read_bytes()+b"\n")
   with self.assertRaisesRegex(LegacyDatasetError,"hash/size"):import_legacy_dataset(a,Path(root)/"out")
  base={"action_label":"rest","action_phase":"rest","confidence":"reported","training_usable_raw":False,"eligible_for_future_training_review":True,"basis":"x"}
  for bounds in ((1,2),(0,3)):
   with self.subTest(bounds=bounds),tempfile.TemporaryDirectory() as root:
    seg=dict(base,start_row=bounds[0],end_row_exclusive=bounds[1]);_,a=self.fixture(root,[self.vec(1)]*2,segments=[seg])
    with self.assertRaisesRegex(LegacyDatasetError,"bounds|cover"):import_legacy_dataset(a,Path(root)/"out")
 def test_overwrite_rejected(self):
  with tempfile.TemporaryDirectory() as root:
   _,a=self.fixture(root,[self.vec(1)]);out=Path(root)/"out";import_legacy_dataset(a,out)
   with self.assertRaises(FileExistsError):import_legacy_dataset(a,out)
 def test_batch_closure_loader_filters_and_groups(self):
  seg=[{"start_row":0,"end_row_exclusive":1,"action_label":"rest","action_phase":"rest","confidence":"reported","training_usable_raw":False,"eligible_for_future_training_review":True,"basis":"x"},{"start_row":1,"end_row_exclusive":2,"action_label":"transition","action_phase":"transition","confidence":"reported","training_usable_raw":False,"eligible_for_future_training_review":False,"basis":"x"}]
  with tempfile.TemporaryDirectory() as root:
   _,out=self.batch(root,first_rows=[self.vec(1),self.vec(2)],first_segments=seg);g=load_experimental_analysis_groups(out/"dataset_manifest.json");self.assertEqual(set(g),{"subject-1"});self.assertEqual(len(g["subject-1"]),7);self.assertEqual(len(g["subject-1"][0].rows),1)
   extra=out/"extra.txt";extra.write_text("x")
   with self.assertRaisesRegex(LegacyDatasetError,"unexpected"):load_experimental_analysis_groups(out/"dataset_manifest.json")
 def test_batch_factor_mapping_and_source_row_tamper_rejected(self):
  for name in ("legacy_write_factor","mapping","source_row_count"):
   with self.subTest(name=name),tempfile.TemporaryDirectory() as root:
    _,out=self.batch(root);dataset=out/"dataset_manifest.json"
    child=out/"session-s1"/"transform_manifest.json";p=json.loads(child.read_text())
    if name=="legacy_write_factor":p["legacy_write_factor"]=2
    elif name=="mapping":p["raw_to_canonical_ranges"][0]["raw_start_row"]=1
    else:p["source"]["row_count"]=2
    child.write_text(json.dumps(p));batch=json.loads(dataset.read_text());ref=batch["sessions"][0]["transform_manifest"];ref["sha256"]=hashlib.sha256(child.read_bytes()).hexdigest();ref["size_bytes"]=child.stat().st_size
    if name=="legacy_write_factor":batch["sessions"][0]["legacy_write_factor"]=2
    dataset.write_text(json.dumps(batch))
    with self.assertRaises(LegacyDatasetError):load_experimental_analysis_groups(dataset)
 def test_resigned_derived_semantic_tamper_is_rejected(self):
  for field,value in (("analysis_scope","artifact_only"),("timestamp","1"),("channel_1","300"),("canonical_index","9")):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
    _,out=self.batch(root);dataset=out/"dataset_manifest.json";csv_path=out/"session-s1"/"derived_samples.csv";child=out/"session-s1"/"transform_manifest.json"
    with csv_path.open(encoding="utf-8",newline="") as stream:reader=csv.DictReader(stream);rows=list(reader);names=reader.fieldnames
    rows[0][field]=value
    with csv_path.open("w",encoding="utf-8",newline="") as stream:w=csv.DictWriter(stream,fieldnames=names,lineterminator="\n");w.writeheader();w.writerows(rows)
    cm=json.loads(child.read_text());cm["output"]["sha256"]=hashlib.sha256(csv_path.read_bytes()).hexdigest();cm["output"]["size_bytes"]=csv_path.stat().st_size;child.write_text(json.dumps(cm))
    bm=json.loads(dataset.read_text());entry=bm["sessions"][0];entry["derived_csv"]["sha256"]=hashlib.sha256(csv_path.read_bytes()).hexdigest();entry["derived_csv"]["size_bytes"]=csv_path.stat().st_size;entry["transform_manifest"]["sha256"]=hashlib.sha256(child.read_bytes()).hexdigest();entry["transform_manifest"]["size_bytes"]=child.stat().st_size;dataset.write_text(json.dumps(bm))
    with self.assertRaises(LegacyDatasetError):load_experimental_analysis_groups(dataset)
 def test_unknown_child_schema_and_missing_child_rejected(self):
  for action in ("schema","missing"):
   with self.subTest(action=action),tempfile.TemporaryDirectory() as root:
    _,out=self.batch(root);child=out/"session-s1"/"transform_manifest.json"
    if action=="schema":p=json.loads(child.read_text());p["version"]="1.2";child.write_text(json.dumps(p))
    else:child.unlink()
    with self.assertRaises(LegacyDatasetError):load_experimental_analysis_groups(out/"dataset_manifest.json")
 def test_scope_is_status_only_and_training_always_rejected(self):
  with tempfile.TemporaryDirectory() as root:
   _,out=self.batch(root,statuses={4:"excluded",5:"completed_non_target"})
   for sid,scope in (("session-s4","excluded"),("session-s5","artifact_only"),("session-s6","diagnostic_only")):
    with (out/sid/"derived_samples.csv").open(encoding="utf-8",newline="") as stream:self.assertTrue(all(row["analysis_scope"]==scope for row in csv.DictReader(stream)))
   with self.assertRaisesRegex(LegacyDatasetError,"not training usable"):load_training_groups(out/"dataset_manifest.json")
 def test_batch_manifest_disagreement_is_atomic(self):
  with tempfile.TemporaryDirectory() as root:
   m,out=self.batch(root);p=json.loads(m.read_text());p["sessions"][0]["legacy_write_factor"]=2;m.write_text(json.dumps(p))
   with self.assertRaisesRegex(LegacyDatasetError,"disagrees"):import_legacy_measurement_manifest(m,Path(root)/"bad")
   self.assertFalse((Path(root)/"bad").exists());self.assertTrue((out/"dataset_manifest.json").exists())
 def test_resigned_alternate_source_and_annotation_cannot_replace_child_identity(self):
  with tempfile.TemporaryDirectory() as root:
   measurement,out=self.batch(root);dataset=out/"dataset_manifest.json";original=Path(root)/"source-1";alternate=Path(root)/"alternate-1";alternate.mkdir()
   (alternate/"client_data.csv").write_bytes((original/"client_data.csv").read_bytes());annotation=json.loads((original/"annotations.json").read_text());annotation["source_directory_name"]="alternate-1";(alternate/"annotations.json").write_text(json.dumps(annotation))
   master=json.loads(measurement.read_text());master["sessions"][0]["relative_directory"]="alternate-1";measurement.write_text(json.dumps(master))
   batch=json.loads(dataset.read_text());batch["source_measurement_manifest"]["sha256"]=hashlib.sha256(measurement.read_bytes()).hexdigest();batch["source_measurement_manifest"]["size_bytes"]=measurement.stat().st_size;entry=batch["sessions"][0]
   for field,name in (("source_csv","client_data.csv"),("annotation","annotations.json")):
    candidate=alternate/name;entry[field]["relative_path"]=str(Path("..")/"alternate-1"/name);entry[field]["sha256"]=hashlib.sha256(candidate.read_bytes()).hexdigest();entry[field]["size_bytes"]=candidate.stat().st_size
   dataset.write_text(json.dumps(batch))
   with self.assertRaisesRegex(LegacyDatasetError,"identity disagrees"):load_experimental_analysis_groups(dataset)

if __name__=="__main__":unittest.main()
