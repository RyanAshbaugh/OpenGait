import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from os import path, listdir, getcwd, makedirs, cpu_count
import argparse
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Queue, Process, Manager

# register xml namespaces
ET.register_namespace("","http://www.nist.gov/briar/xml/media")
ET.register_namespace("media","http://www.nist.gov/briar/xml/media")
ET.register_namespace("vc","http://www.w3.org/2007/XMLSchema-versioning")
ET.register_namespace("cmn","http://standards.iso.org/iso-iec/39794/-1")
ET.register_namespace("vstd","http://standards.iso.org/iso-iec/30137/-4")
ET.register_namespace("fstd","http://standards.iso.org/iso-iec/39794/-5")
ET.register_namespace("wb","http://standards.iso.org/iso-iec/39794/-16")
ET.register_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")

ns = {"vstd":"http://standards.iso.org/iso-iec/30137/-4",
        "fstd":"http://standards.iso.org/iso-iec/39794/-5"}

def parallelXMLParse( video_path, process_label_df_dict, index, progress_bar ):

    # BiometricModality values
    # whole body, face
    wb_biomodality, face_biomodality = '15', '1'

    biometric_modalities = { face_biomodality: "face", wb_biomodality: "wb" }
    wb_labels_df_columns = [ 'frame_index', 'x', 'y', 'w', 'h' ]
    face_labels_df_columns = [ 'frame_index', 'x', 'y', 'w', 'h','yaw','pitch' ]

    wb_labels_df = pd.DataFrame(columns=wb_labels_df_columns)
    face_labels_df = pd.DataFrame(columns=face_labels_df_columns)

    labels_dict = { wb_biomodality: wb_labels_df, face_biomodality: face_labels_df }

    tree = ET.parse(video_path)
    root = tree.getroot()

    for frame_annot in root.iterfind('.//vstd:FrameAnnotation', ns ):
        frame_index = int( frame_annot.findall('.//vstd:FrameIndex',ns)[0].text )
        for object_annot in frame_annot.iterfind('.//vstd:ObjectAnnotation', ns ):
            modality = object_annot.findall('.//vstd:BiometricModality',ns)[0].text
            x = int( object_annot.findall('.//vstd:x',ns)[0].text )
            y = int( object_annot.findall('.//vstd:y',ns)[0].text )
            w = int( object_annot.findall('.//vstd:boxWidth',ns)[0].text )
            h = int( object_annot.findall('.//vstd:boxHeight',ns)[0].text )
            label_row_dict = { 'frame_index': frame_index,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h }
            if biometric_modalities[modality] == "face":
                yaw = float( object_annot.findall('.//fstd:yawAngleBlock/fstd:angleValue',ns)[0].text )
                pitch = float( object_annot.findall('.//fstd:pitchAngleBlock/fstd:angleValue',ns)[0].text )
                label_row_dict['yaw'] = yaw
                label_row_dict['pitch'] = pitch

            # add the label, whether face or whole body, to the correct dictionary
            labels_dict[modality] = labels_dict[modality].append( label_row_dict, ignore_index = True )

    process_label_df_dict[str(index)] = { 'wb_labels':labels_dict['15'], 'face_labels':labels_dict['1'] }
    progress_bar.put(1)

def errorCallback(e):
    print(e)

def progressBarListener(q, num_iters):
    progress_bar = tqdm(total = num_iters, desc='Generating BRIAR dataset dataframe')
    for item in iter(q.get, None):
        progress_bar.update()

class BRIARDataset():
    def __init__( self, dataset_path, dataset_pickle_path='' ):
        self.dataset_path = dataset_path
        full_labels_df_columns = [ 'path',
                                    'fname',
                                    'controlled/field',
                                    'stand/struct/rand',
                                    'wb/face',
                                    'set_num',
                                    'subject_id',
                                    'full/distractor',
                                    'distance/uav',
                                    'labels_path',
                                    'wb_labels',
                                    'face_labels' ]
        self.full_label_df = pd.DataFrame( columns=full_labels_df_columns )

        self.prepareVideoDataset(dataset_pickle_path)

    def getVideoNames( self ):

        # if the dataset is loaded with the precomputed pickle file, update the
        # video file paths, otherwise put them together from scratch
        if self.full_label_df.shape[0] > 0:
            for ii, fname in enumerate(self.full_label_df['fname']):
                path_name = path.join(self.dataset_path,
                                         self.full_label_df['full/distractor'][ii],
                                         self.full_label_df['subject_id'][ii],
                                         self.full_label_df['controlled/field'][ii],
                                         self.full_label_df['distance/uav'][ii],
                                         self.full_label_df['wb/face'][ii],
                                         self.full_label_df['fname'][ii]+'.mp4')
                self.full_label_df['path'][ii] = path_name

        else:
            self.full_label_df['path'] = [ ii for ii in Path( self.dataset_path ).rglob('*.mp4') ]
            self.full_label_df['fname'] = [ ii.stem for ii in self.full_label_df['path'] ]

        return

    def getVideoMetaData( self ):

        for ii, video_path in enumerate( self.full_label_df['path'] ):

            video_info = str(video_path).split('/')
            wb_face = video_info[-2]
            distance_uav = video_info[-3]
            controlled_field = video_info[-4]
            subject_id = video_info[-5]

            specific_info = self.full_label_df['fname'][ii].split('/')[-1].split('_')
            set_num = specific_info[1]
            stand_struct_rand = specific_info[2]

            self.full_label_df['controlled/field'][ii] = controlled_field
            self.full_label_df['wb/face'][ii] = wb_face
            self.full_label_df['stand/struct/rand'][ii] = stand_struct_rand
            self.full_label_df['set_num'][ii] = set_num
            self.full_label_df['subject_id'][ii] = subject_id
            self.full_label_df['distance/uav'][ii] = distance_uav

            # set the full/distractor value, relevant for test, so see if the
            # name has either in it already
            self.full_label_df['full/distractor'][ii] = ''
            if 'full' in str(video_path):
                self.full_label_df['full/distractor'][ii] = 'full'
            elif 'distractor' in str(video_path):
                self.full_label_df['full/distractor'][ii] = 'distractor'

            # some label xml files use detections at the end of the name instead of tracks
            label_path_name = str(self.full_label_df['path'][ii])[:-4] + '_WB_face_tracks.xml'
            if not path.exists( label_path_name ):
                label_path_name = str(self.full_label_df['path'][ii])[:-4] + '_WB_face_detections.xml'

            self.full_label_df['labels_path'][ii] = label_path_name

    def loadVideoLabels( self ):

        pool = Pool(processes=cpu_count()-4)

        #num_videos = 10
        num_videos = len(self.full_label_df['labels_path'])

        manager = Manager()
        progress_bar_queue = manager.Queue()
        progress_bar_listener = Process(target=progressBarListener, args=(progress_bar_queue,num_videos))
        progress_bar_listener.start()

        # set up the dataframe to be shared between processes
        process_label_df_dict = manager.dict()

        # for each video, parse the corresponding xml
        for ii, video_path in enumerate( self.full_label_df['labels_path'][0:num_videos] ):

            # parallel
            pool.apply_async( parallelXMLParse,args=[video_path, process_label_df_dict, ii, progress_bar_queue],\
                error_callback=errorCallback )

        pool.close()
        pool.join()

        progress_bar_queue.put(None)
        progress_bar_listener.join()

        for key,val in process_label_df_dict.items():
            self.full_label_df['wb_labels'][int(key)] = val['wb_labels']
            self.full_label_df['face_labels'][int(key)] = val['face_labels']

    def prepareVideoDataset( self, dataset_pickle_path ):

        if path.exists(dataset_pickle_path):
            with open(dataset_pickle_path,'rb') as pkl_file:
                self.full_label_df = pickle.load( pkl_file )
                pkl_file.close()
            self.getVideoNames()
        else:
            response = input('Dataset dataframe {} not found.'.format(dataset_pickle_path) + '\nContinue '
                             'with generating new dataframe? y/n: ')
            if response == 'y':
                self.getVideoNames()
                self.getVideoMetaData()
                self.loadVideoLabels()
                #briar_pickle_fname = path.join(getcwd(),'briar_dataset.pkl')
                if not path.exists(dataset_pickle_path):
                    briar_pickle = open(dataset_pickle_path, 'wb')
                    pickle.dump(self.full_label_df, briar_pickle)
                    briar_pickle.close()
            else:
                print('Not generating new dataframe, goodbye')
                sys.exit()
        return

    def getVideoIndicesByCondition(self, distance, condition,
                                   controlled_field='field'):
        cont_field_indice_series = self.full_label_df.index[
            (self.full_label_df['controlled/field'] == controlled_field)]
        distance_indice_series = self.full_label_df.index[
            (self.full_label_df['distance/uav'] == distance[0])]
        condition_indice_series = self.full_label_df.index[
            (self.full_label_df['stand/struct/rand'] == condition[0])]

        selected_indices = cont_field_indice_series.intersection(
            distance_indice_series.intersection(condition_indice_series))
        return selected_indices
