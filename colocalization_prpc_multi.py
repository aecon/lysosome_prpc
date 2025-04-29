import os
import csv
import numpy as np
from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed

import colocalization_prpc_single as CPS


def process_image_multi(paths, shared_list, lock):

    total_prpc_nucleus, total_prpc_cytoplasm, total_prpc_celledge, total_prpc_lysosome, cellArea, nucleiArea, Nnuclei = CPS.process_image(paths)

    # get row, column from file name
    well, field = paths[3].split("POM2")[1].split("_")[0].split("-")
    row = well[0]
    col = int(well[1::])

    result = {
        "File": os.path.basename(paths[4]),
        "Well": well,
        "Row": row,
        "Column": col,
        "CellArea": cellArea,
        "NucleiArea": nucleiArea,
        "Ncells": Nnuclei,
        "PrPc_in_DAPI": total_prpc_nucleus,
        "PrPc_in_Cytoplasm": total_prpc_cytoplasm,
        "PrPc_in_CellEdge": total_prpc_celledge,
        "PrPc_in_Lysosome": total_prpc_lysosome
    }

    with lock:
        shared_list.append(result)


if __name__ == "__main__":

    plate_name = "plate3"
    files_nuc, files_cel, files_raw_cel, files_lys, files_prp = CPS.get_path_lists(plate_name)
    Nfiles = len(files_nuc)

    max_workers = 64

    output_file = "output_PrPc_localization_%s.csv" % plate_name

    with Manager() as manager:
        shared_list = manager.list()
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image_multi, paths, shared_list, lock)
                for paths in zip(files_nuc, files_cel, files_raw_cel, files_lys, files_prp)
            ]
            for i,future in enumerate(as_completed(futures), 1):
                future.result()
                if i%10==0 or i==Nfiles:
                    print(f"Processed {i}/{Nfiles} images")

        header = ["File", "Well", "Row", "Column",
                    "CellArea", "NucleiArea",
                    "Ncells",
                    "PrPc_in_DAPI", "PrPc_in_Cytoplasm", "PrPc_in_CellEdge", "PrPc_in_Lysosome"]

        with open(output_file, 'w', newline='\n') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in shared_list:
                writer.writerow(row)

