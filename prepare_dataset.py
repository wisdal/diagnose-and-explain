# Copyright 2018 Wisdom D'Almeida
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import tensorflow as tf
import xml.etree.ElementTree
import numpy as np

image_folder = ''
annotation_folder = ''

def maybe_download():
    image_zip = 'NLMCXR_png.tgz'
    if not os.path.exists(os.path.abspath('.') + '/dataset/' + image_zip):
        tf.keras.utils.get_file('NLMCXR_png.tgz',
                                cache_subdir=os.path.abspath('.') + '/dataset/',
                                origin = 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz',
                                extract = True)
        tf.keras.utils.get_file('NLMCXR_reports.tgz',
                                cache_subdir=os.path.abspath('.') + '/dataset/',
                                origin = 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz',
                                extract = True)

    global image_folder, annotation_folder
    image_folder = './dataset/'
    annotation_folder = './dataset/ecgen-radiology/'

def extract_data():
    all_findings = []
    all_impressions = []
    all_img_names = []
    rids = []

    total_count = 0 # Count of reports available in the dataset
    no_image_count = 0 # Count of reports having no associated chest image
    no_impression_count = 0 # Count of reports having an empty "Impression" section
    no_findings_count = 0 # Count of reports having an empty "Findings" section

    # Storing impressions, findings and the image names in vectors
    for file in os.listdir(annotation_folder):
        total_count += 1
        file = os.path.abspath(annotation_folder) + '/' + file
        e = xml.etree.ElementTree.parse(file).getroot()

        rid = e.find('pmcId').get('id') # Report Id
        # We choose to ignore reports having no associated image
        image_id = e.find('parentImage')
        if image_id is None:
            no_image_count += 1
            continue

        image_id = image_id.get('id')
        image_name = os.path.abspath('.') + '/' + image_id + '.png'
        findings = ''
        impression = ''

        # Parsing "Impression" and "Findings"
        for element in e.findall('MedlineCitation/Article/Abstract/AbstractText'):
            if element.get('Label') == 'FINDINGS':
                findings = element.text
            if element.get('Label') == 'IMPRESSION':
                impression = element.text

        # Sanity check: Skip this report if it has an empty "Impression" section
        if findings is None:
            no_findings_count += 1
            #findings = 'No finding'
            continue
        if impression is None:
            no_impression_count += 1
            continue

        # Transforming findings and impressions into lists of sentences
        findings = findings.replace("XXXX", "") #"XXXX" represents information anonymized
        sentences = findings.split('.')
        del sentences[-1]
        sentences = ['<start> ' + sentence + ' <end>' for sentence in sentences]
        findings = sentences

        impression = impression.replace("XXXX", "") #"XXXX" represents information anonymized
        sentences = impression.split('.')
        del sentences[-1]
        sentences = ['<start> ' + sentence + ' <end>' for sentence in sentences]
        impression = sentences

        #appending to vectors
        all_img_names.append(image_name)
        all_findings.append(findings)
        all_impressions.append(impression)
        rids.append(rid)

    print("Number of reports available:", total_count)
    print("Number of reports selected:", len(all_img_names))
    print("Number of reports not having images (skipped):", no_image_count)
    print("Number of reports with Impression section empty (skipped):", no_impression_count)
    print("Number of reports with Findings section empty:", no_findings_count)
    print("Total skipped:", no_image_count + no_impression_count + no_findings_count)

    return all_findings, all_impressions, all_img_names, rids

def init_inception_model():
    # Initialize InceptionV3 and load the pretrained Imagenet weights
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    return tf.keras.Model(new_input, hidden_layer)

def transform_input(all_findings, all_impressions, max_paragraph_length, max_sentence_length):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    findings_texts = [' '.join(findings) for findings in all_findings]
    impressions_texts = [' '.join(impression) for impression in all_impressions]
    tokenizer.fit_on_texts(findings_texts + impressions_texts)
    all_findings_seq = [tokenizer.texts_to_sequences(findings) for findings in all_findings]
    all_impressions_seq = [tokenizer.texts_to_sequences(impression) for impression in all_impressions]

    tokenizer.word_index['<pad>'] = 0

    # Adding empty sentence seqs to each paragraph to have a fixed length for each
    for findings in all_findings_seq:
        while len(findings) < max_paragraph_length:
            findings.append([0])
        if len(findings) > max_paragraph_length:
            del findings[max_paragraph_length:]

    for impressions in all_impressions_seq:
        while len(impressions) < max_paragraph_length:
            impressions.append([0])
        if len(impressions) > max_paragraph_length:
            del impressions[max_paragraph_length:]

    # Padding sequences
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    findings_vector = [pad_sequences(findings, padding='post', maxlen=max_sentence_length) for findings in all_findings_seq]
    impressions_vector = [pad_sequences(impressions, padding='post', maxlen=max_sentence_length) for impressions in all_impressions_seq]

    #print(findings_vector[0])
    #print(impressions_vector[0])

    # Combining findings and impressions
    for i in range(len(findings_vector)):
        findings_vector[i] = np.concatenate((findings_vector[i], impressions_vector[i]))
    # Now, for a given batch "i", we can retrieve impressions = findings[i, max_paragraph_length:]
    #print(findings_vector[0])

    return tokenizer, findings_vector