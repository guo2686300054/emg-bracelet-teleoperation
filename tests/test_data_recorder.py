import csv,inspect,json,os,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest import mock
from data_recorder import DataRecorder,RecorderCloseError,RecorderQualitySnapshot,generate_subject_key,recover_incomplete_sessions,validate_session_id,validate_subject_key
from emg_protocol import AcquisitionMetadata,DeviceKey,EmgFrame,HandSide,NotificationPacketProtocol,PADDED28_PROTOCOL,QualityFlags,RateDescriptor,SequenceTracker,SignalChain,from_shared_v2_flags,to_shared_v2_flags,validate_quality_flags
from recording_context import RecordingContext
import session_quality
SUB="sub-0123456789abcdef0123456789abcdef"
DEV=DeviceKey("dev-0123456789abcdef0123456789abcdef")
BASE_FLAGS=QualityFlags.VALID|QualityFlags.HOST_WALL_TIME_VALID|QualityFlags.HOST_MONOTONIC_VALID|QualityFlags.HOST_RECEIVE_INDEX_VALID
class RecorderTests(unittest.TestCase):
 def make(self,d,**kw):
  args=dict(subject_id=SUB,device_id=DEV,acquisition=AcquisitionMetadata(device_sequence_modulus=1<<64,device_sequence_modulus_source="wire_uint64_contract",signal_chain=SignalChain(sample_format="float32")),channels=2,session_id="session-a",flush_every=1,side=HandSide.LEFT);args.update(kw)
  if "recording_context" not in args:
   side=args["side"];args["recording_context"]=RecordingContext(args["subject_id"],"rest","hold","test_v1",side) if side in {HandSide.LEFT,HandSide.RIGHT} else None
  return DataRecorder(d,**args)
 def write(self,r,i=0,**kw):
  values=tuple(kw.pop("channel_values",(1,2)));default_sequence=i if r._acquisition_metadata["device_sequence_modulus"] is not None else None;args=dict(host_wall_timestamp_ns=100+i,host_monotonic_ns=200+i,host_receive_index=i,sample_index=i,device_packet_sequence=default_sequence,sample_in_packet=0,action_label="rest",action_phase="hold");args.update(kw);flags=args.pop("quality_flags",BASE_FLAGS);sequence=args.get("device_packet_sequence");
  if isinstance(flags,QualityFlags) and sequence is not None:flags|=QualityFlags.DEVICE_PACKET_SEQUENCE_VALID
  r.record(EmgFrame(values,quality_flags=flags,**args))
 def staged_recovery(self,d,count=2,**kwargs):
  r=self.make(d,**kwargs)
  for index in range(count):self.write(r,index)
  r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
  return r
 def mutate_csv_row(self,r,row_index,key,value):
  with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:
   reader=csv.DictReader(stream);rows=list(reader);fieldnames=reader.fieldnames
  rows[row_index][key]=value
  with r.csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
   writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
 def assert_invalid_recovery_preserves_metadata(self,d,r,detail):
  original=r.metadata_path.read_bytes();report=recover_incomplete_sessions(d)
  self.assertEqual(len(report.recovered),0);self.assertEqual(len(report.issues),1);self.assertIn(detail,report.issues[0].detail);self.assertEqual(r.metadata_path.read_bytes(),original)
  return report
 def downgrade_to_legacy_identity(self,r,device_id):
  payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload["schema_minor"]=0;payload["device_id"]=device_id;r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
  with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
  for row in rows:row.pop("generation",None);row.pop("connection_generation",None);row.pop("session_sample_index",None);row["device_id"]=device_id
  with r.csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
   writer=csv.DictWriter(stream,fieldnames=list(DataRecorder.LEGACY_BASE_COLUMNS)+[f"channel_{index}" for index in range(1,r.channels+1)]);writer.writeheader();writer.writerows(rows)
 def make_historical_fixture(self,r,schema_minor,remove_sequence_contract=True):
  payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload["schema_minor"]=schema_minor
  if schema_minor<4:
   payload.pop("hand_side",None);payload.pop("writer_generation_semantics",None);payload.pop("connection_generation_semantics",None)
  if schema_minor<3:payload["acquisition"]["signal_chain"].pop("quantization")
  if schema_minor<5:payload["acquisition"].pop("notification_packet_protocol")
  if schema_minor<7:payload.get("session_boundary",{}).pop("host_monotonic_tie_count",None)
  if schema_minor<8:payload.pop("extra",None)
  if remove_sequence_contract:
   payload["acquisition"].pop("device_sequence_modulus");payload["acquisition"].pop("device_sequence_modulus_source")
  identity="legacy-device-v0" if schema_minor==0 else str(DEV);payload["device_id"]=identity
  r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
  with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
  for row in rows:
   if schema_minor<6:row.pop("session_sample_index",None)
   if schema_minor<2:row.pop("generation",None)
   if schema_minor<4:row.pop("connection_generation",None)
   row["device_id"]=identity
   if remove_sequence_contract:
    row["device_packet_sequence"]="";row["quality_flags"]=str(int(QualityFlags(int(row["quality_flags"]))&~(QualityFlags.DEVICE_PACKET_SEQUENCE_VALID|QualityFlags.DUPLICATE_PACKET|QualityFlags.OUT_OF_ORDER_PACKET|QualityFlags.PACKET_GAP)))
  if schema_minor>=6:base=DataRecorder.BASE_COLUMNS
  elif schema_minor>=4:base=DataRecorder.V1_5_BASE_COLUMNS
  elif schema_minor>=2:base=DataRecorder.V1_2_BASE_COLUMNS
  else:base=DataRecorder.LEGACY_BASE_COLUMNS
  columns=base+tuple(f"channel_{index}" for index in range(1,r.channels+1))
  with r.csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
   writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
 def clear_csv_device_sequences(self,r):
  with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:
   reader=csv.DictReader(stream);rows=list(reader);columns=reader.fieldnames
  for row in rows:
   row["device_packet_sequence"]="";row["quality_flags"]=str(int(QualityFlags(int(row["quality_flags"]))&~(QualityFlags.DEVICE_PACKET_SEQUENCE_VALID|QualityFlags.DUPLICATE_PACKET|QualityFlags.OUT_OF_ORDER_PACKET|QualityFlags.PACKET_GAP)))
  with r.csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
   writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
 def test_subject_key_and_extra_whitelist(self):
  self.assertRegex(generate_subject_key(),r"^sub-[0-9a-f]{32}$");validate_subject_key(SUB)
  for bad in ("Alice","sub-XYZ","sub-"+"a"*31):
   with self.assertRaises(ValueError):validate_subject_key(bad)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):self.make(d,metadata_extra={"patient_name":"Alice"})
 def test_public_session_id_validator_is_the_recorder_contract(self):
  self.assertEqual(validate_session_id("CaseName"),"casename")
  for value in ("../escape","COM1","name:stream","trail.","",None):
   with self.subTest(value=value),self.assertRaises(ValueError):validate_session_id(value)
 def test_strict_contract_and_monotonicity(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0)
   for changes in (dict(host_receive_index=0),dict(sample_index=0),dict(quality_flags="VALID"),dict(host_wall_timestamp_ns="now")):
    with self.subTest(changes=changes),self.assertRaises(ValueError):self.write(r,1,**changes)
   self.write(r,2,sample_index=5,device_packet_sequence=2);r.close()
   with r.csv_path.open(encoding="utf-8-sig") as stream: rows=list(csv.DictReader(stream))
   self.assertEqual(len(rows),2);self.assertEqual(rows[0]["quality_flags"],str(int(BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID)))
 def test_late_session_accepts_global_index_origin_and_reports_exact_regression(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,3500);self.write(r,3501)
   cases=((dict(host_receive_index=3499,sample_index=3502,host_monotonic_ns=3702),"host_receive_index moved backwards"),(dict(host_receive_index=3502,sample_index=3501,host_monotonic_ns=3702),"sample_index is not strictly increasing"),(dict(host_receive_index=3502,sample_index=3502,host_monotonic_ns=3699),"host_monotonic_ns moved backwards"))
   for changes,message in cases:
    with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):self.write(r,3502,**changes)
   r.close()
   with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
   self.assertEqual([int(row["host_receive_index"]) for row in rows],[3500,3501]);self.assertEqual([int(row["sample_index"]) for row in rows],[3500,3501])
 def test_checkpoint_and_acquisition_metadata(self):
  acq=AcquisitionMetadata(signal_chain=SignalChain(sample_format="uint8",unit="adc_code",adc_bit_width=8))
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,acquisition=acq);self.write(r)
   m=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(m["last_persisted_row"],1);self.assertEqual(m["status"],"recording");self.assertEqual(m["schema_id"],"emg.session.metadata");self.assertEqual(m["acquisition"]["signal_chain"]["filter"],"unknown");r.close()
 def test_notification_packet_protocol_is_persisted_and_recovery_validates_it(self):
  protocol=NotificationPacketProtocol(PADDED28_PROTOCOL,28,16,"zero_suffix","probe:sha256")
  acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format="uint8"),notification_packet_protocol=protocol)
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1,acquisition=acquisition);payload=json.loads(r.metadata_path.read_text(encoding="utf8"))
   self.assertEqual(payload["schema_minor"],8);self.assertEqual(payload["acquisition"]["notification_packet_protocol"],{"mode":PADDED28_PROTOCOL,"wire_packet_size":28,"logical_packet_size":16,"padding_rule":"zero_suffix","evidence_ref":"probe:sha256"})
   payload["acquisition"]["notification_packet_protocol"]["wire_packet_size"]=27;r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"packet protocol contract")
 def test_forced_exit_recovery_reconciles_rows(self):
  with tempfile.TemporaryDirectory() as d:
   code=("from data_recorder import *;from recording_context import *;from emg_protocol import *;c=RecordingContext('"+SUB+"','rest','hold','test_v1',HandSide.LEFT);r=DataRecorder(r'"+d.replace("\\","\\\\")+"',subject_id='"+SUB+"',device_id=DeviceKey('"+str(DEV)+"'),acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format='float32')),channels=1,flush_every=1,side=HandSide.LEFT,recording_context=c);r.record(EmgFrame((1,),host_wall_timestamp_ns=1,host_monotonic_ns=1,host_receive_index=0,sample_index=0,action_label='rest',action_phase='hold'));import os;os._exit(7)")
   result=subprocess.run([sys.executable,"-c",code]);self.assertEqual(result.returncode,7)
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.recovered[0].csv_rows,1)
   m=json.loads((report.recovered[0].session_dir/"metadata.json").read_text(encoding="utf8"));self.assertEqual(m["status"],"incomplete");self.assertTrue(m["recovery"]["consistent"])
 def test_protocol_vector_unknown_bits_and_typed_rate(self):
  self.assertEqual(to_shared_v2_flags(QualityFlags.VALID|QualityFlags.CRC_ERROR),1025)
  with self.assertRaises(ValueError):validate_quality_flags(QualityFlags(1<<30))
  with self.assertRaises(ValueError):RateDescriptor(50,"unknown","x",True)
 def test_protocol_has_no_transport_import_and_adapter_vector_is_fixed(self):
  code="import sys,emg_protocol;assert 'shared_memory_v2' not in sys.modules;print(emg_protocol.to_shared_v2_flags(emg_protocol.QualityFlags.VALID|emg_protocol.QualityFlags.HOST_WALL_TIME_VALID|emg_protocol.QualityFlags.STALE|emg_protocol.QualityFlags.DUPLICATE_PACKET))"
  result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True)
  self.assertEqual(result.stdout.strip(),str(1+4+2048))
  semantic=QualityFlags.VALID|QualityFlags.HOST_WALL_TIME_VALID|QualityFlags.HOST_MONOTONIC_VALID|QualityFlags.HOST_RECEIVE_INDEX_VALID|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID|QualityFlags.DEVICE_SAMPLE_COUNTER_VALID|QualityFlags.DEVICE_TIME_VALID|QualityFlags.DISCONNECTED|QualityFlags.OVERFLOW|QualityFlags.CRC_ERROR|QualityFlags.STALE
  self.assertEqual(to_shared_v2_flags(semantic),4093);self.assertEqual(from_shared_v2_flags(4093),semantic)
  with self.assertRaises(ValueError):from_shared_v2_flags(2)
 def test_sequence_tracker_is_bounded_and_handles_wrap_gap_reorder_generation(self):
  tracker=SequenceTracker(256);self.assertEqual(tracker.observe(254).flags,QualityFlags.NONE);self.assertEqual(tracker.observe(255).flags,QualityFlags.NONE);self.assertEqual(tracker.observe(0).flags,QualityFlags.NONE)
  gap=tracker.observe(2);self.assertEqual(gap.flags,QualityFlags.PACKET_GAP);self.assertEqual(gap.gap_count,1);self.assertEqual(tracker.observe(2).flags,QualityFlags.DUPLICATE_PACKET);self.assertEqual(tracker.observe(1).flags,QualityFlags.OUT_OF_ORDER_PACKET);self.assertEqual(tracker.observe(130).flags,QualityFlags.OUT_OF_ORDER_PACKET);self.assertEqual(tracker.observe(1,generation=1).flags,QualityFlags.NONE)
  size_before=sys.getsizeof(tracker)
  for value in range(1_000_000):tracker.observe(value%256,generation=2)
  self.assertFalse(hasattr(tracker,"__dict__"));self.assertEqual(len(tracker.checkpoint()),2);self.assertEqual(sys.getsizeof(tracker),size_before)
 def test_public_frame_is_constructed_once_and_recorder_computes_sequence_flags(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);frame=EmgFrame((1,2),100,200,0,0,device_packet_sequence=9,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID);r.record(frame)
   duplicate=EmgFrame((1,2),101,201,1,1,device_packet_sequence=9,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID);r.record(duplicate);r.close()
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   self.assertFalse(hasattr(r,"record_sample"));self.assertTrue(int(rows[1]["quality_flags"])&int(QualityFlags.DUPLICATE_PACKET))
 def test_sample_formats_ranges_validity_and_tick_provenance(self):
  with self.assertRaises(ValueError):SignalChain(sample_format="uint8",adc_bit_width=12)
  narrowed=SignalChain(sample_format="uint8",adc_bit_width=12,quantization="right_shift_4_then_clip")
  self.assertEqual(narrowed.adc_bit_width,12);self.assertEqual(narrowed.sample_format,"uint8")
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,channels=1,acquisition=AcquisitionMetadata(signal_chain=narrowed));r.record(EmgFrame((255,),1,1,0,0,action_label="rest",action_phase="hold"))
   with self.assertRaises(ValueError):r.record(EmgFrame((256,),2,2,1,1,action_label="rest",action_phase="hold"))
   r.close()
  cases=(("uint8",(0,),True),("uint8",(-1,),False),("uint8",(1.5,),False),("int16",(-32768,),True),("int16",(32768,),False),("float32",(1.25,),True),("float32",(3.5e38,),False),("float32",(True,),False))
  for index,(sample_format,values,valid) in enumerate(cases):
   with self.subTest(sample_format=sample_format,values=values),tempfile.TemporaryDirectory() as d:
    r=self.make(d,channels=1,session_id=f"format-{index}",acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format=sample_format)))
    if valid:r.record(EmgFrame(values,1,1,0,0,action_label="rest",action_phase="hold"))
    else:
     with self.assertRaises(ValueError):r.record(EmgFrame(values,1,1,0,0,action_label="rest",action_phase="hold"))
    r.close()
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,channels=1);frame=EmgFrame((1,),1,1,0,0,device_time_ticks=1,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_TIME_VALID)
   with self.assertRaises(ValueError):r.record(frame)
   r.close()
  with tempfile.TemporaryDirectory() as d:
   acquisition=AcquisitionMetadata(device_tick_rate=RateDescriptor(1000,"firmware","rev1",True),device_tick_modulus=65536,device_tick_modulus_source="firmware_rev1",signal_chain=SignalChain(sample_format="uint8"));r=self.make(d,channels=1,acquisition=acquisition);r.record(EmgFrame((1,),1,1,0,0,device_time_ticks=1,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_TIME_VALID));
   with self.assertRaises(ValueError):r.record(EmgFrame((1,),2,2,1,1,device_time_ticks=65536,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_TIME_VALID))
   r.close()
 def test_optional_validity_and_sequence_flags_are_not_trusted(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d)
   for frame in (EmgFrame((1,2),1,1,0,0,quality_flags=QualityFlags.VALID),EmgFrame((1,2),1,1,0,0,device_packet_sequence=1,quality_flags=BASE_FLAGS),EmgFrame((1,2),1,1,0,0,quality_flags=BASE_FLAGS|QualityFlags.DUPLICATE_PACKET)):
    with self.assertRaises(ValueError):r.record(frame)
   r.close()
 def test_duplicate_packet_is_audited_with_new_host_index(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=7)
   self.write(r,1,device_packet_sequence=7);r.close()
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   self.assertEqual([x["host_receive_index"] for x in rows],["0","1"]);self.assertTrue(int(rows[1]["quality_flags"])&int(QualityFlags.DUPLICATE_PACKET))
 def test_quality_snapshot_counts_only_trusted_device_sequence_results(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=10);self.write(r,1,device_packet_sequence=10);self.write(r,2,device_packet_sequence=13);self.write(r,3,device_packet_sequence=12)
   snapshot=r.quality_snapshot();self.assertIsInstance(snapshot,RecorderQualitySnapshot);self.assertTrue(snapshot.sequence_detection_available);self.assertEqual((snapshot.duplicate_count,snapshot.out_of_order_count,snapshot.gap_count),(1,1,2));self.assertEqual(snapshot.connection_generation,0);self.assertEqual(snapshot.recorded_rows,4)
   with self.assertRaises((AttributeError,TypeError)):snapshot.recorded_rows=99
   r.close()
 def test_quality_snapshot_marks_unknown_sequence_unavailable_without_payload_guessing(self):
  acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format="float32"))
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,acquisition=acquisition);self.write(r,0,channel_values=(7,7));self.write(r,1,channel_values=(7,7));snapshot=r.quality_snapshot()
   self.assertFalse(snapshot.sequence_detection_available);self.assertEqual((snapshot.duplicate_count,snapshot.out_of_order_count,snapshot.gap_count),(0,0,0));self.assertEqual(snapshot.recorded_rows,2);r.close()
 def test_quality_snapshot_is_atomic_while_recording(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=1000);snapshots=[r.quality_snapshot()];finished=threading.Event()
   def observe():
    while not finished.is_set():snapshots.append(r.quality_snapshot())
   thread=threading.Thread(target=observe);thread.start()
   try:
    for index in range(40):self.write(r,index)
   finally:finished.set();thread.join(timeout=2);r.close()
   self.assertFalse(thread.is_alive());self.assertTrue(snapshots);self.assertTrue(all(0<=item.recorded_rows<=40 for item in snapshots));self.assertEqual(r.quality_snapshot().recorded_rows,40)
 def test_failed_write_does_not_commit_sequence_quality_counters(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=1000);self.write(r,0,device_packet_sequence=7)
   with mock.patch.object(r._writer,"writerow",side_effect=OSError("write failed")):
    with self.assertRaises(OSError):self.write(r,1,device_packet_sequence=7)
   snapshot=r.quality_snapshot();self.assertEqual(snapshot.recorded_rows,1);self.assertEqual(snapshot.duplicate_count,0);r.close()
 def test_multi_sample_packet_reuses_host_index(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=3,sample_in_packet=0);self.write(r,1,host_receive_index=0,host_wall_timestamp_ns=100,host_monotonic_ns=200,device_packet_sequence=3,sample_in_packet=1);r.close()
 def test_false_duplicate_mark_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d)
   try:
    with self.assertRaises(ValueError):self.write(r,0,device_packet_sequence=9,quality_flags=QualityFlags.DUPLICATE_PACKET)
   finally:r.close()
 def test_session_unique(self):
  with tempfile.TemporaryDirectory() as d:
   a=self.make(d);a.close();b=self.make(d);b.close();self.assertNotEqual(a.session_id,b.session_id)
 def test_active_process_is_not_recovered(self):
  with tempfile.TemporaryDirectory() as d:
   ready=Path(d)/"ready";code=("from pathlib import Path;from data_recorder import *;from recording_context import *;from emg_protocol import *;c=RecordingContext('"+SUB+"','rest','hold','test_v1',HandSide.LEFT);r=DataRecorder(r'"+d.replace("\\","\\\\")+"',subject_id='"+SUB+"',device_id=DeviceKey('"+str(DEV)+"'),acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format='float32')),channels=1,side=HandSide.LEFT,recording_context=c);Path(r'"+str(ready).replace("\\","\\\\")+"').write_text('ready');import time;time.sleep(10)")
   process=subprocess.Popen([sys.executable,"-c",code])
   try:
    import time
    for _ in range(100):
     if ready.exists():break
     time.sleep(.01)
    self.assertTrue(ready.exists());report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),0);self.assertEqual(report.issues[0].code,"active")
   finally: process.terminate();process.wait(timeout=3)
 def test_recovery_rejects_future_schema_and_traversal(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);r.close();m=json.loads(r.metadata_path.read_text(encoding="utf8"));m["status"]="recording";m["schema_minor"]=999;m["csv_file"]="../x.csv";r.metadata_path.write_text(json.dumps(m),encoding="utf8")
   original=r.metadata_path.read_bytes();report=recover_incomplete_sessions(d);self.assertEqual(report.issues[0].code,"invalid_session");self.assertEqual(r.metadata_path.read_bytes(),original)
 def test_deepcopy_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   extra={"firmware_version":"fw-1.0"};r=self.make(d,metadata_extra=extra);extra["firmware_version"]="fw-2.0";self.assertEqual(json.loads(r.metadata_path.read_text(encoding="utf8"))["extra"]["firmware_version"],"fw-1.0");r.close()
 def test_recording_context_is_persisted_and_enforced_on_csv_frames(self):
  with tempfile.TemporaryDirectory() as d:
   context=RecordingContext(SUB,"fist","hold","discrete_hand_v1",HandSide.LEFT)
   r=self.make(d,side=HandSide.LEFT,recording_context=context)
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["extra"],context.to_metadata_extra());self.assertNotIn("name",json.dumps(payload).casefold())
   self.write(r,0,action_label="fist",action_phase="hold")
   with self.assertRaisesRegex(ValueError,"metadata action context"):
    self.write(r,1,action_label="rest",action_phase="hold")
   r.close()
 def test_recording_context_identity_and_side_must_match_recorder(self):
  with tempfile.TemporaryDirectory() as d:
   wrong_subject=RecordingContext("sub-ffffffffffffffffffffffffffffffff","rest","hold","study_v1",HandSide.LEFT)
   with self.assertRaisesRegex(ValueError,"subject_id"):
    self.make(d,side=HandSide.LEFT,recording_context=wrong_subject)
   context=RecordingContext(SUB,"rest","hold","study_v1",HandSide.RIGHT)
   with self.assertRaisesRegex(ValueError,"hand_side"):
    self.make(d,side=HandSide.LEFT,recording_context=context)
 def test_metadata_extra_cannot_be_a_second_action_or_privacy_entrypoint(self):
  invalid=({"action_label":"rest"},{"action_phase":"hold"},{"experiment_id":"study_v1"},{"patient_name":"Alice"})
  with tempfile.TemporaryDirectory() as d:
   for index,extra in enumerate(invalid):
    with self.subTest(extra=extra),self.assertRaises(ValueError):self.make(d,session_id=f"invalid-{index}",metadata_extra=extra)
 def test_metadata_versions_are_bounded_safe_ascii_scalars(self):
  with tempfile.TemporaryDirectory() as d:
   invalid=({},[],["v1"],"line\nbreak","has space","固件1","x"*97,1,None)
   for index,value in enumerate(invalid):
    with self.subTest(value=value),self.assertRaises(ValueError):self.make(d,session_id=f"invalid-version-{index}",metadata_extra={"firmware_version":value})
   r=self.make(d,session_id="valid-version",metadata_extra={"firmware_version":"fw-1.2_3","protocol_version":"ble-v1"});r.close()
 def test_new_recording_requires_context_and_known_hand(self):
  with tempfile.TemporaryDirectory() as d:
   parameter=inspect.signature(DataRecorder).parameters["recording_context"]
   self.assertIs(parameter.kind,inspect.Parameter.KEYWORD_ONLY);self.assertIs(parameter.default,inspect.Parameter.empty)
   with self.assertRaisesRegex(ValueError,"recording_context is required"):
    self.make(d,recording_context=None)
   with self.assertRaisesRegex(ValueError,"HandSide.LEFT"):
    self.make(d,side=HandSide.UNKNOWN,recording_context=RecordingContext(SUB,"rest","hold","study_v1",HandSide.LEFT))
 def test_recovery_rejects_metadata_csv_action_disagreement_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["extra"]["action_label"]="fist";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"CSV action annotation")
 def test_schema_17_recovery_keeps_historical_action_compatibility(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["schema_minor"]=7;payload["extra"]={};r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   recovered=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(recovered["schema_minor"],7);self.assertEqual(recovered["extra"],{})
   quality=session_quality.analyze_session(r.session_dir);self.assertFalse(quality["training_usable"])
 def test_bare_acquisition_dict_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):self.make(d,acquisition={"sample_rate":50})
 def test_exact_csv_header(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,channels=3);r.close()
   with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:header=next(csv.reader(stream))
   self.assertEqual(header,list(DataRecorder.BASE_COLUMNS)+["channel_1","channel_2","channel_3"])
 def test_rate_descriptor_schema_rejects_bool_negative_string_and_unknown(self):
  for args in ((True,"x","y",False),(-1,"x","y",False),("50","x","y",False),(50," unknown ","y",True),(50,"x"," unknown ",True),(50,"x","y","true")):
   with self.subTest(args=args),self.assertRaises(ValueError):RateDescriptor(*args)
 def test_all_quality_flag_values_are_frozen(self):
  expected={"NONE":0,"VALID":1,"HOST_WALL_TIME_VALID":4,"HOST_MONOTONIC_VALID":8,"HOST_RECEIVE_INDEX_VALID":16,"DEVICE_PACKET_SEQUENCE_VALID":32,"DEVICE_SAMPLE_COUNTER_VALID":64,"DEVICE_TIME_VALID":128,"DISCONNECTED":256,"OVERFLOW":512,"CRC_ERROR":1024,"STALE":2048,"DUPLICATE_PACKET":65536,"OUT_OF_ORDER_PACKET":131072,"PACKET_GAP":262144}
  self.assertEqual({name:int(getattr(QualityFlags,name)) for name in expected},expected)
 def test_same_packet_requires_identical_timestamps(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=3)
   self.write(r,1,host_receive_index=0,host_wall_timestamp_ns=100,host_monotonic_ns=200,device_packet_sequence=3,sample_in_packet=1)
   with self.assertRaises(ValueError):self.write(r,2,host_receive_index=0,host_wall_timestamp_ns=101,host_monotonic_ns=200,device_packet_sequence=3,sample_in_packet=2)
   r.close()
 def test_duplicate_multi_sample_packet_keeps_duplicate_flag(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=7)
   self.write(r,1,device_packet_sequence=7)
   self.write(r,2,host_receive_index=1,host_wall_timestamp_ns=101,host_monotonic_ns=201,device_packet_sequence=7,sample_in_packet=1)
   r.close()
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   self.assertEqual([row["sample_in_packet"] for row in rows],["0","0","1"])
 def test_aborted_terminal_intent_survives_lock_release_retry(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r)
   original=r._session_lock.release;calls=0
   def flaky_release():
    nonlocal calls
    calls+=1
    if calls==1:raise OSError("lock release")
    return original()
   with mock.patch.object(r._session_lock,"release",side_effect=flaky_release):
    with self.assertRaises(RecorderCloseError):r.close(complete=False,error="device disconnected")
    r.close()
   metadata=json.loads(r.metadata_path.read_text(encoding="utf8"))
   self.assertEqual(metadata["status"],"aborted");self.assertEqual(metadata["abort_reason"],"device disconnected")
 def test_writer_failure_latches_fault_and_close_is_deterministic(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=10)
   with mock.patch.object(r._writer,"writerow",side_effect=OSError("write failed")):
    with self.assertRaises(OSError):self.write(r)
   live=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(live["status"],"faulted");self.assertEqual(live["fault"]["stage"],"writerow")
   r.close();r.close()
   final=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(final["status"],"faulted")
 def test_periodic_flush_failure_latches_fault_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=1)
   with mock.patch.object(r,"_sync_csv",side_effect=OSError("flush failed")):
    with self.assertRaises(OSError):self.write(r)
   metadata=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(metadata["fault"]["stage"],"flush")
   r.close()
 def test_close_flush_failure_is_reported_then_closes_faulted(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=10);self.write(r)
   with mock.patch.object(r,"_sync_csv",side_effect=OSError("close flush")):
    with self.assertRaises(RecorderCloseError) as caught:r.close()
   self.assertEqual(caught.exception.failures[0][0],"close_flush")
   r.close();metadata=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(metadata["status"],"faulted");self.assertEqual(metadata["fault"]["stage"],"close_flush")
 def test_fault_metadata_failure_is_audited_without_hiding_primary(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=10);original=r._write_metadata;calls=0
   def flaky_metadata(status):
    nonlocal calls
    calls+=1
    if calls==1:raise PermissionError("metadata denied")
    return original(status)
   with mock.patch.object(r,"_write_metadata",side_effect=flaky_metadata),mock.patch.object(r._writer,"writerow",side_effect=OSError("writer primary")):
    with self.assertRaisesRegex(OSError,"writer primary"):self.write(r)
   self.assertEqual(r._failures[0]["stage"],"writerow");self.assertEqual(r._failures[1]["stage"],"fault_metadata")
   r.close()
 def test_file_close_failure_retries_and_preserves_fault(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r);original=r._close_csv;calls=0
   def flaky_close():
    nonlocal calls
    calls+=1
    if calls==1:raise OSError("file close")
    return original()
   with mock.patch.object(r,"_close_csv",side_effect=flaky_close):
    with self.assertRaises(RecorderCloseError):r.close()
    r.close()
   metadata=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(metadata["status"],"faulted");self.assertEqual(metadata["fault"]["stage"],"file_close")
 def test_terminal_metadata_commit_failure_retries_as_faulted(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r);original=r._replace_metadata;calls=0
   def flaky_replace(path):
    nonlocal calls
    calls+=1
    if calls in {2,3}:raise PermissionError("metadata commit")
    return original(path)
   with mock.patch.object(r,"_replace_metadata",side_effect=flaky_replace):
    with self.assertRaises(RecorderCloseError) as caught:r.close()
    self.assertEqual(caught.exception.failures[0][0],"metadata_commit")
    r.close()
   metadata=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(metadata["status"],"faulted");self.assertEqual(metadata["fault"]["stage"],"metadata_commit")
 def test_recovery_schema_types_are_rejected_independently_and_scan_continues(self):
  invalid_values=(("schema_major",True),("schema_minor",-1),("schema_minor","0"))
  for key,value in invalid_values:
   with self.subTest(key=key,value=value),tempfile.TemporaryDirectory() as d:
    bad=self.make(d,session_id="bad");bad.close();payload=json.loads(bad.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload[key]=value;bad.metadata_path.write_text(json.dumps(payload),encoding="utf8")
    original=bad.metadata_path.read_bytes()
    good=self.make(d,session_id="good");good.close();valid=json.loads(good.metadata_path.read_text(encoding="utf8"));valid["status"]="recording";good.metadata_path.write_text(json.dumps(valid),encoding="utf8")
    report=recover_incomplete_sessions(d);self.assertEqual(len(report.issues),1);self.assertEqual(len(report.recovered),1);self.assertEqual(bad.metadata_path.read_bytes(),original)
 def test_recovery_streams_csv_and_uses_valid_rows_as_terminal_fact(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=5000)
   for index in range(2000):self.write(r,index)
   r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload["row_count"]=0;payload["last_persisted_row"]=0;r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   report=recover_incomplete_sessions(d);self.assertEqual(report.recovered[0].csv_rows,2000);final=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual((final["row_count"],final["last_persisted_row"]),(2000,2000));self.assertFalse(final["recovery"]["consistent"]);self.assertEqual(final["recovery"]["strategy"],"validated_csv_rows_are_fact")
 def test_corrupt_partial_csv_is_not_committed(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";original=json.dumps(payload,sort_keys=True);r.metadata_path.write_text(original,encoding="utf8")
   with r.csv_path.open("a",encoding="utf8") as stream:stream.write("1,2\n")
   report=recover_incomplete_sessions(d);self.assertEqual(report.issues[0].code,"invalid_session");self.assertEqual(r.metadata_path.read_text(encoding="utf8"),original)
 def test_recovery_replace_and_lock_release_failures_are_issues_and_scan_continues(self):
  import data_recorder as module
  with tempfile.TemporaryDirectory() as d:
   sessions=[]
   for name in ("one","two"):
    r=self.make(d,session_id=name);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8");sessions.append(r)
   original_replace=module._replace_recovery_metadata;replace_calls=0
   def flaky_replace(temp,target):
    nonlocal replace_calls
    replace_calls+=1
    if replace_calls==1:raise PermissionError("replace")
    return original_replace(temp,target)
   with mock.patch.object(module,"_replace_recovery_metadata",side_effect=flaky_replace):report=recover_incomplete_sessions(d)
   self.assertEqual(len(report.recovered),1);self.assertTrue(any(issue.code=="metadata_commit" for issue in report.issues))
  with tempfile.TemporaryDirectory() as d:
   for name in ("one","two"):
    r=self.make(d,session_id=name);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   original_release=module.SessionLock.release;release_calls=0
   def flaky_release(lock):
    nonlocal release_calls
    original_release(lock);release_calls+=1
    if release_calls==1:raise OSError("release")
   with mock.patch.object(module.SessionLock,"release",flaky_release):report=recover_incomplete_sessions(d)
   self.assertEqual(len(report.recovered),2);self.assertTrue(any(issue.code=="lock_release" for issue in report.issues))
 def test_recovery_traversal_and_ads_are_independent(self):
  for csv_name in ("../samples.csv","samples.csv:secret"):
   with self.subTest(csv_name=csv_name),tempfile.TemporaryDirectory() as d:
    r=self.make(d);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload["csv_file"]=csv_name;r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
    original=r.metadata_path.read_bytes();report=recover_incomplete_sessions(d);self.assertEqual(len(report.issues),1);self.assertEqual(report.issues[0].code,"invalid_session");self.assertEqual(r.metadata_path.read_bytes(),original)
 def test_windows_alias_identifiers_are_rejected_and_case_collides(self):
  with tempfile.TemporaryDirectory() as d:
   for session in ("trail.","COM1","COM1.txt","name:stream"):
    with self.subTest(session=session),self.assertRaises(ValueError):self.make(d,session_id=session)
   upper=self.make(d,session_id="CaseName");upper.close();lower=self.make(d,session_id="casename");lower.close();self.assertEqual(upper.session_id,"casename");self.assertEqual(lower.session_id,"casename-001")
 def test_subject_junction_or_symlink_escape_is_rejected(self):
  with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
   subject=Path(root)/SUB
   if os.name=="nt":
    result=subprocess.run(["cmd.exe","/c","mklink","/J",str(subject),str(Path(outside))],capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr or result.stdout)
   else:subject.symlink_to(Path(outside),target_is_directory=True)
   with self.assertRaises(ValueError):self.make(root)
 def test_recovery_never_enters_external_junction_or_touches_lock(self):
  with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
   external_session=Path(outside)/"session";external_session.mkdir();metadata=external_session/"metadata.json";original='{"outside":true}';metadata.write_text(original,encoding="utf8")
   junction=Path(root)/"linked-subject"
   if os.name=="nt":
    result=subprocess.run(["cmd.exe","/c","mklink","/J",str(junction),str(Path(outside))],capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr or result.stdout)
   else:junction.symlink_to(Path(outside),target_is_directory=True)
   report=recover_incomplete_sessions(root)
   self.assertEqual(len(report.recovered),0);self.assertTrue(any(issue.code=="unsafe_path" for issue in report.issues))
   self.assertEqual(metadata.read_text(encoding="utf8"),original);self.assertFalse((external_session/".session.lock").exists())
 def test_constructor_preserves_primary_and_aggregates_cleanup_failures(self):
  primary=OSError("initial sync primary")
  original_close=DataRecorder._close_csv;original_release=__import__("data_recorder").SessionLock.release
  def close_then_fail(recorder):
   original_close(recorder);raise PermissionError("csv cleanup")
  def release_then_fail(lock):
   original_release(lock);raise TimeoutError("lock cleanup")
  with tempfile.TemporaryDirectory() as d,mock.patch.object(DataRecorder,"_sync_csv",side_effect=primary),mock.patch.object(DataRecorder,"_close_csv",close_then_fail),mock.patch("data_recorder.SessionLock.release",release_then_fail):
   with self.assertRaises(OSError) as caught:self.make(d)
  self.assertIs(caught.exception,primary);self.assertEqual([stage for stage,_ in caught.exception.cleanup_failures],["close_csv","release_session_lock"]);self.assertEqual(caught.exception.initialization_resource_state,"CLOSE_FAILED");self.assertFalse(caught.exception.initialization_lock_released)
 def test_record_and_close_are_serialized_without_partial_row(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,flush_every=10);original=r._writer;entered=threading.Event();release=threading.Event();record_done=threading.Event();close_done=threading.Event();errors=[]
   class BlockingWriter:
    def writerow(self,row):
     entered.set()
     if not release.wait(1):raise TimeoutError("test writer barrier")
     return original.writerow(row)
   r._writer=BlockingWriter()
   def record():
    try:self.write(r)
    except BaseException as exc:errors.append(exc)
    finally:record_done.set()
   def close():
    try:r.close()
    except BaseException as exc:errors.append(exc)
    finally:close_done.set()
   writer_thread=threading.Thread(target=record);closer_thread=threading.Thread(target=close)
   writer_thread.start();self.assertTrue(entered.wait(1));closer_thread.start();time.sleep(.05);self.assertFalse(close_done.is_set())
   with r.csv_path.open(encoding="utf-8-sig") as stream:self.assertEqual(len(list(csv.DictReader(stream))),0)
   release.set();self.assertTrue(record_done.wait(1));self.assertTrue(close_done.wait(1));writer_thread.join(1);closer_thread.join(1);self.assertFalse(writer_thread.is_alive());self.assertFalse(closer_thread.is_alive());self.assertEqual(errors,[])
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   metadata=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(len(rows),1);self.assertEqual(metadata["row_count"],1);self.assertEqual(metadata["last_persisted_row"],1);self.assertEqual(metadata["status"],"complete")
   with self.assertRaises(RuntimeError):self.write(r,1)
 def test_recovery_rejects_missing_required_quality_flag(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);self.mutate_csv_row(r,0,"quality_flags",str(int(QualityFlags.HOST_WALL_TIME_VALID|QualityFlags.HOST_MONOTONIC_VALID|QualityFlags.HOST_RECEIVE_INDEX_VALID|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID)))
   self.assert_invalid_recovery_preserves_metadata(d,r,"validity flags")
 def test_recovery_rejects_optional_field_flag_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);self.mutate_csv_row(r,0,"device_packet_sequence","")
   self.assert_invalid_recovery_preserves_metadata(d,r,"optional field")
 def test_recovery_rejects_metadata_path_identity_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["session_id"]="another-session";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"session identity")
 def test_recovery_rejects_csv_identity_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);self.mutate_csv_row(r,0,"device_id","dev-ffffffffffffffffffffffffffffffff")
   self.assert_invalid_recovery_preserves_metadata(d,r,"CSV identity")
 def test_recovery_rejects_cross_row_sample_regression(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d);self.mutate_csv_row(r,1,"sample_index","0")
   self.assert_invalid_recovery_preserves_metadata(d,r,"sample_index")
 def test_recovery_rejects_nonconsecutive_samples_in_packet(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=3);self.write(r,1,host_receive_index=0,host_wall_timestamp_ns=100,host_monotonic_ns=200,device_packet_sequence=3,sample_in_packet=1);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8");self.mutate_csv_row(r,1,"sample_in_packet","2")
   self.assert_invalid_recovery_preserves_metadata(d,r,"packet identity")
 def test_recovery_rejects_sequence_flag_disagreement(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=7);self.write(r,1,device_packet_sequence=7);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8");self.mutate_csv_row(r,1,"quality_flags",str(int(BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID)))
   self.assert_invalid_recovery_preserves_metadata(d,r,"SequenceTracker")
 def test_recovery_rejects_rate_descriptor_disagreement(self):
  with tempfile.TemporaryDirectory() as d:
   acquisition=AcquisitionMetadata(sample_rate=RateDescriptor(200,"protocol","spec",True),signal_chain=SignalChain(sample_format="float32"));r=self.staged_recovery(d,count=1,acquisition=acquisition);self.mutate_csv_row(r,0,"sample_rate_hz","201")
   self.assert_invalid_recovery_preserves_metadata(d,r,"sample rate differs")
 def test_recovery_rejects_old_minor_unknown_sample_format_without_guessing(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload["schema_minor"]=0;payload["acquisition"]["signal_chain"]["sample_format"]="unknown";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"explicit sample_format")
 def test_recovery_uses_persisted_generation_for_sequence_reset(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=7,generation=10,connection_generation=0);self.write(r,1,device_packet_sequence=7,generation=10,connection_generation=1);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
 def test_minor_zero_known_format_accepts_exact_legacy_identity(self):
  with tempfile.TemporaryDirectory() as d:
   acquisition=AcquisitionMetadata(device_sequence_modulus=256,device_sequence_modulus_source="legacy_protocol",signal_chain=SignalChain(sample_format="float32"));r=self.staged_recovery(d,count=1,acquisition=acquisition);self.downgrade_to_legacy_identity(r,"legacy-device-A")
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertTrue(payload["recovery"]["legacy_identity"]);self.assertEqual(payload["device_id"],"legacy-device-A")
 def test_minor_zero_legacy_identity_rejects_control_characters(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);self.downgrade_to_legacy_identity(r,"legacy\ndevice")
   self.assert_invalid_recovery_preserves_metadata(d,r,"legacy device identity")
 def test_minor_zero_legacy_identity_must_match_csv_exactly(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);self.downgrade_to_legacy_identity(r,"Legacy-Device");self.mutate_csv_row(r,0,"device_id","legacy-device")
   self.assert_invalid_recovery_preserves_metadata(d,r,"CSV identity")
 def test_minor_one_and_newer_require_opaque_device_key(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["device_id"]="legacy-device";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"device key")
 def test_unknown_sequence_modulus_rejects_sequence_but_allows_unsequenced_data(self):
  with tempfile.TemporaryDirectory() as d:
   acquisition=AcquisitionMetadata(signal_chain=SignalChain(sample_format="uint8"));r=self.make(d,channels=1,acquisition=acquisition)
   with self.assertRaisesRegex(ValueError,"configured modulus"):
    r.record(EmgFrame((1,),1,1,0,0,device_packet_sequence=1,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID))
   r.record(EmgFrame((1,),2,2,0,0,action_label="rest",action_phase="hold"));r.close()
  with self.assertRaises(ValueError):AcquisitionMetadata(device_sequence_modulus=256)
  with self.assertRaises(ValueError):AcquisitionMetadata(device_sequence_modulus_source="claimed")
 def test_config_to_recorder_wrap_and_recovery_end_to_end(self):
  from app_config import load_config
  with tempfile.TemporaryDirectory() as d:
   config_path=Path(d)/"config.ini";config_path.write_text("[App]\nconfig_version=1.1\n[BLE]\ndevice_id="+str(DEV)+"\n[Data]\nchannels=1\ndata_path=data\nsample_format=uint8\nadc_bit_width=12\nquantization=right_shift_4_then_clip\ndevice_sequence_modulus=256\ndevice_sequence_modulus_source=protocol_spec_v1\n",encoding="utf8")
   config=load_config(config_path);metadata=config.to_acquisition_metadata();self.assertEqual(metadata.device_sequence_modulus,256);self.assertEqual(metadata.signal_chain.adc_bit_width,12)
   context=RecordingContext(SUB,"rest","hold","test_v1",HandSide.LEFT);r=DataRecorder(config.data_path,subject_id=SUB,device_id=config.ble.device_key,acquisition=metadata,channels=1,session_id="wrap",flush_every=1,side=HandSide.LEFT,recording_context=context)
   for index,value in enumerate((254,255,0)):
    r.record(EmgFrame((value,),index+1,index+1,index,index,device_packet_sequence=value,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID))
   r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   self.assertTrue(all(not int(row["quality_flags"])&int(QualityFlags.OUT_OF_ORDER_PACKET) for row in rows))
   report=recover_incomplete_sessions(config.data_path);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
 def test_real_historical_acquisition_fixtures_are_migrated_without_guessing(self):
  for schema_minor in (0,1,2):
   with self.subTest(schema_minor=schema_minor),tempfile.TemporaryDirectory() as d:
    r=self.staged_recovery(d,count=2);self.make_historical_fixture(r,schema_minor)
    report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
    payload=json.loads(r.metadata_path.read_text(encoding="utf8"));migrations=payload["recovery"]["metadata_migrations"]
    self.assertEqual({item["field"] for item in migrations},{"acquisition.device_sequence_modulus","acquisition.device_sequence_modulus_source","acquisition.signal_chain.quantization","acquisition.notification_packet_protocol"})
    self.assertEqual({item["field"] for item in payload["recovery"]["session_migrations"]},{"hand_side","connection_generation","session_boundary.host_monotonic_tie_count"})
    self.assertIsNone(payload["acquisition"]["device_sequence_modulus"]);self.assertEqual(payload["acquisition"]["device_sequence_modulus_source"],"unknown");self.assertEqual(payload["acquisition"]["signal_chain"]["quantization"],"unknown")
 def test_historical_explicit_sequence_contract_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   acquisition=AcquisitionMetadata(device_sequence_modulus=256,device_sequence_modulus_source="legacy_protocol",signal_chain=SignalChain(sample_format="float32"));r=self.staged_recovery(d,count=2,acquisition=acquisition);self.make_historical_fixture(r,2,remove_sequence_contract=False)
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["acquisition"]["device_sequence_modulus"],256);self.assertEqual([item["field"] for item in payload["recovery"]["metadata_migrations"]],["acquisition.signal_chain.quantization","acquisition.notification_packet_protocol"])
 def test_schema_thirteen_requires_exact_current_acquisition_fields(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["acquisition"].pop("device_sequence_modulus");r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"schema 1.5 acquisition fields")
 def test_historical_partial_sequence_contract_is_invalid_and_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=1);self.make_historical_fixture(r,2);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["acquisition"]["device_sequence_modulus"]=256;r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"requires provenance")
 def test_schema_twelve_container_width_default_is_downgraded_without_sequence(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=2);self.make_historical_fixture(r,2,remove_sequence_contract=False);self.clear_csv_device_sequences(r)
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertIsNone(payload["acquisition"]["device_sequence_modulus"]);self.assertEqual(payload["acquisition"]["device_sequence_modulus_source"],"unknown")
   removed=[item for item in payload["recovery"]["metadata_migrations"] if item["action"]=="removed_historical_container_width_assumption"]
   self.assertEqual({item["field"] for item in removed},{"acquisition.device_sequence_modulus","acquisition.device_sequence_modulus_source"});self.assertEqual({item["previous_value"] for item in removed},{1<<64,"wire_uint64_contract"})
 def test_schema_twelve_container_default_with_sequence_is_not_certified(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=2);self.make_historical_fixture(r,2,remove_sequence_contract=False)
   self.assert_invalid_recovery_preserves_metadata(d,r,"configured modulus")
 def test_hand_side_is_strict_and_persisted_in_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d,side=HandSide.LEFT);payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["hand_side"],"left");self.assertNotIn("hand_side",DataRecorder.BASE_COLUMNS);r.close()
   with self.assertRaises(ValueError):self.make(d,session_id="bad-side",side="left")
  with self.assertRaises(ValueError):HandSide("左")
 def test_connection_generation_resets_sequence_without_false_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=7,generation=3,connection_generation=10);self.write(r,1,device_packet_sequence=7,generation=3,connection_generation=11);r.close()
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   self.assertEqual([row["connection_generation"] for row in rows],["10","11"]);self.assertFalse(int(rows[1]["quality_flags"])&int(QualityFlags.DUPLICATE_PACKET))
 def test_writer_generation_does_not_reset_connection_sequence_tracker(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,device_packet_sequence=7,generation=3,connection_generation=10);self.write(r,1,device_packet_sequence=7,generation=4,connection_generation=10);r.close()
   with r.csv_path.open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
   self.assertTrue(int(rows[1]["quality_flags"])&int(QualityFlags.DUPLICATE_PACKET))
 def test_connection_generation_must_not_move_backwards(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,connection_generation=2)
   with self.assertRaisesRegex(ValueError,"connection_generation"):
    self.write(r,1,connection_generation=1)
   r.close()
 def test_recovery_rejects_connection_generation_regression_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,connection_generation=1);self.write(r,1,connection_generation=2);r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8");self.mutate_csv_row(r,1,"connection_generation","0")
   self.assert_invalid_recovery_preserves_metadata(d,r,"connection_generation")
 def test_session_sample_index_is_continuous_and_independent_of_global_index(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,3500)
   with self.assertRaisesRegex(ValueError,"session_sample_index is not continuous"):
    frame=EmgFrame((1,2),101,201,3501,3501,device_packet_sequence=3501,action_label="rest",action_phase="hold",quality_flags=BASE_FLAGS|QualityFlags.DEVICE_PACKET_SEQUENCE_VALID)
    r.record(frame,session_sample_index=2)
   self.write(r,3501);r.close()
   with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
   self.assertEqual([int(row["sample_index"]) for row in rows],[3500,3501]);self.assertEqual([int(row["session_sample_index"]) for row in rows],[0,1])

 def test_equal_monotonic_timestamps_are_recorded_and_audited(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,host_monotonic_ns=500);self.write(r,1,host_monotonic_ns=500);r.close()
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["session_boundary"]["host_monotonic_tie_count"],1)
   with r.csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
   self.assertEqual([int(row["host_monotonic_ns"]) for row in rows],[500,500]);self.assertEqual([int(row["host_receive_index"]) for row in rows],[0,1])

 def test_high_frequency_monotonic_ties_keep_index_order_and_reject_regression(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d)
   for index in range(100):self.write(r,index,host_monotonic_ns=1000+index//4)
   with self.assertRaisesRegex(ValueError,"host_monotonic_ns moved backwards"):
    self.write(r,100,host_monotonic_ns=1023)
   r.close();payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["session_boundary"]["host_monotonic_tie_count"],75)

 def test_recovery_recomputes_and_validates_monotonic_tie_count(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,host_monotonic_ns=700);self.write(r,1,host_monotonic_ns=700);r.close()
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   recovered=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(recovered["session_boundary"]["host_monotonic_tie_count"],1)
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,host_monotonic_ns=700);self.write(r,1,host_monotonic_ns=700);r.close()
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));payload["status"]="recording";payload["session_boundary"]["host_monotonic_tie_count"]=0;r.metadata_path.write_text(json.dumps(payload),encoding="utf8")
   self.assert_invalid_recovery_preserves_metadata(d,r,"tie count")

 def test_schema_six_recovery_computes_monotonic_ties_without_rejecting_them(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);self.write(r,0,host_monotonic_ns=900);self.write(r,1,host_monotonic_ns=900);r.close();self.make_historical_fixture(r,6,remove_sequence_contract=False)
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["session_boundary"]["host_monotonic_tie_count"],1);self.assertEqual(payload["recovery"]["source_schema_minor"],6);self.assertEqual(payload["recovery"]["target_schema_minor"],8)
   migration=next(item for item in payload["recovery"]["session_migrations"] if item["field"]=="session_boundary.host_monotonic_tie_count");self.assertEqual(migration["value"],1)

 def test_schema_six_recovery_still_rejects_monotonic_regression(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=2);self.make_historical_fixture(r,6,remove_sequence_contract=False);self.mutate_csv_row(r,1,"host_monotonic_ns","199")
   self.assert_invalid_recovery_preserves_metadata(d,r,"host monotonic time moved backwards")

 def test_session_boundary_audit_is_persisted(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make(d);r.configure_session_boundary(start_host_receive_index=20,start_drop_total=3);self.write(r,21)
   r.update_session_boundary(end_host_receive_index=22,received_count=2,eligible_count=1,written_count=1,queue_drop_total=4,queue_drop_session=1,tail_pending_count=0,tail_loss_count=1,incomplete_reason="queue_drop");r.close()
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["session_boundary"],{"start_host_receive_index":20,"end_host_receive_index":22,"received_count":2,"eligible_count":1,"written_count":1,"queue_drop_total":4,"queue_drop_session":1,"tail_pending_count":0,"tail_loss_count":1,"host_monotonic_tie_count":0,"incomplete_reason":"queue_drop"})

 def test_recovery_rejects_noncontinuous_session_sample_index(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=2);self.mutate_csv_row(r,1,"session_sample_index","0")
   self.assert_invalid_recovery_preserves_metadata(d,r,"session_sample_index")

 def test_schema_three_csv_migrates_unknown_connection_generation(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.staged_recovery(d,count=2);self.make_historical_fixture(r,3,remove_sequence_contract=False)
   report=recover_incomplete_sessions(d);self.assertEqual(len(report.recovered),1);self.assertEqual(report.issues,[])
   payload=json.loads(r.metadata_path.read_text(encoding="utf8"));self.assertEqual(payload["hand_side"],"unknown");self.assertEqual(payload["connection_generation_semantics"],"unknown_assumed_zero_for_validation");self.assertEqual([item["field"] for item in payload["recovery"]["metadata_migrations"]],["acquisition.notification_packet_protocol"])
if __name__=="__main__":unittest.main()
