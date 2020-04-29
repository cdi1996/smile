# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np

import json
from collections import OrderedDict

def scan_known_people(known_people_folder): # 아는 사람의 얼굴과 이름을 배열에 append
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0] #경로 중 파일 명만 얻어옴
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img) #

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))
            known_names.append(basename)
            known_face_encodings.append(encodings[0])
        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0]) #배열에 넣어주어 리턴

    print(encodings[0])
    return known_names, known_face_encodings

def upload_unknown_file(upload_file): #업로드된 파일들 검사 후 배열에 저장
    upload_image = face_recognition.load_image_file(upload_file)

    if(max(upload_image.shape) > 1600):
        pil_img = PIL.Image.fromarray(upload_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS) # 크기 줄임
        upload_image = np.array(pil_img)

    upload_encodings = face_recognition.face_encodings(upload_image)
    #print(upload_encodings)

    f=open('unknown_encodings_save.json', 'a', encoding="utf-8")
    json.dump(, indent="\t")
    np.savetxt(f, upload_encodings, delimiter=',', newline='\n')
    f.close()


def print_result(filename, name, distance, show_distance=False):
    if show_distance:
        print("{},{},{}".format(filename, name, distance))
    else:
        print("{},{}".format(filename, name))

"""그러면
파일이 하나씩 들어올 때마다 upload_btn_click로 파일 개개마다 검사하여 인코딩값 저장
그리고 가장 마지막에 테스트 이미지를 실행시키도록
"""


def selfie_upload_btn(selfie_file, user_id):
    img = face_recognition.load_image_file(selfie_file)
    user_encodings = face_recognition.face_encodings(img)

    if len(user_encodings) > 1:
        click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))
    if len(user_encodings) == 0:
        click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))

    # user_id path 처리


    compare_image(selfie_file, user_id, user_encodings, 0.3, False)


def compare_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):

    user_faces = []

    f=open('unknown_encoding_save.txt', 'r')
    # load할 파일이 비었을?
    unknown_encodings = np.loadtxt(f, delimiter=',')
    f.close()

    n=len(unknown_encodings)

    if not unknown_encodings.any():
        # print out fact that no faces were found in image
        print_result(image_to_check, "no_persons_found", None, show_distance)

    if n!=128: # 사진 한명 이상일 경우.....
        for unknown_encoding in unknown_encodings:
            distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
            result = list(distances <= tolerance)

            if True in result:
                user_faces.append(picture_name)
                [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
            else:
                print_result(image_to_check, "unknown_person", None, show_distance)
    else:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encodings)
        result = list(distances <= tolerance)

        print("----------------------")
        print(unknown_encodings)

        if True in result:
            [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
        else:
            print_result(image_to_check, "unknown_person", None, show_distance)



def image_files_in_folder(folder): # pwd 효과
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.argument('known_people_folder')
@click.argument('image_to_check')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
def main(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    # known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)


if __name__ == "__main__":
    main()
