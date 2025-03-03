import os
import re

def update_sh_files(directory):
    # 遍历指定目录及其子目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sh"):  # 确保处理的是.sh文件
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for line in lines:
                        # if '"patience": 3, ' in line:
                        #     # 替换指定的字符串
                        #     line = line.replace('"patience": 3, ', '')
                        # if 'Few' in line:
                        #     # 替换指定的字符串
                        #     line = line.replace('Few', 'FEW')
                        # if 'Full' in line:
                        #     # 替换指定的字符串
                        #     line = line.replace('Full', 'FULL')
                        # if 'Zero' in line:
                        #     # 替换指定的字符串
                        #     line = line.replace('Zero', 'ZERO')
                        # if 'Moirai' in line:
                        #     line = line.replace('"patience": 3, ', '')
                            
                        if 'Moirai' in line or 'TinyTimeMixer' in line or 'RoseModel' in line:
                            # 替换指定的字符串
                            # line = line.replace('"patience": 3, ', '')
                            # 替换指定的字符串
                            # 替换指定的字符串
                            path = re.search(r'--save-path "([^"]+)"', line)
                            if path:
                                paths = path.group(1).split('/')
                                
                            line = line.replace(f'"{path.group(1)}"', f'"{paths[1]}/{paths[2]}/{paths[0]}"')
                        # if 'gpus 4' in line:
                        #     line = line.replace('gpus 4', 'gpus 0')
                        # if 'gpus 3' in line:
                        #     line = line.replace('gpus 3', 'gpus 0')
                        # if 'gpus 2' in line:
                        #     line = line.replace('gpus 2', 'gpus 0')
                        # if 'gpus 1' in line:
                        #     line = line.replace('gpus 1', 'gpus 0')
                        # if 'gpus 5' in line:
                        #     line = line.replace('gpus 5', 'gpus 0')
                        # if 'gpus 6' in line:
                        #     line = line.replace('gpus 6', 'gpus 0')
                        # if 'gpus 7' in line:
                        #     line = line.replace('gpus 7', 'gpus 0')
                        # if 'TinyTimeMixer' in line:
                        #     if '"horizon": 192' in line:
                        #         line = line.replace('"horizon": 96', '"horizon": 192')
                        #     elif '"horizon": 336' in line:
                        #         line = line.replace('"horizon": 96', '"horizon": 336')
                        #     elif '"horizon": 720' in line:
                        #         line = line.replace('"horizon": 96', '"horizon": 720')
                        # if 'RoseModel' in line:
                        #     if ', "sample_rate": 0.05' in line:
                        #         line = line.replace(', "sample_rate": 0.05', ', "sample_rate": 1')
                        #     if 'FEW' in line:
                        #         line = line.replace('FEW', 'FULL')
                        #     if '"seq_len": 128':
                        #         line = line.replace('"seq_len": 128', '"seq_len": 512')
                        #     if '"horizon": 36' in line:
                        #         line = line.replace('"horizon": 36', '"horizon": 192')
                        #         line = line.replace('"horizon": 96', '"horizon": 192')
                        #     elif '"horizon": 48' in line:
                        #         line = line.replace('"horizon": 48', '"horizon": 336')
                        #         line = line.replace('"horizon": 96', '"horizon": 336')
                        #     elif '"horizon": 60' in line:
                        #         line = line.replace('"horizon": 60', '"horizon": 720')
                        #         line = line.replace('"horizon": 96', '"horizon": 720')
                        #     elif '"horizon": 24' in line:
                                # line = line.replace('"horizon": 24', '"horizon": 96')
                        # if 'TimesFM' in line:
                            # if '"horizon": 36' in line or '"horizon": 48' in line or '"horizon":60' in line:
                            #     line = line.replace(', "is_train": 1', ', "is_train": 0, "get_train": 1')
                            #     line = line.replace('"horizon": 192', '"horizon": 96')
                            #     line = line.replace('"horizon": 60', '"horizon": 96')
                            #     line = line.replace('"horizon": 60', '"horizon": 96')
                            # if '"horizon": 96' in line:
                            #     line = line.replace(', "is_train": 0, "get_train": 1', ', "is_train": 1')
                            # if '"horizon": 192' in line:
                            #     line = line.replace('--model-hyper-params \'{"horizon": 192', '--model-hyper-params \'{"horizon": 96')
                            # if '"horizon": 336' in line:
                            #     line = line.replace('--model-hyper-params \'{"horizon": 336', '--model-hyper-params \'{"horizon": 96')
                                
                            # if '"horizon": 720' in line:
                            #     line = line.replace('--model-hyper-params \'{"horizon": 720', '--model-hyper-params \'{"horizon": 96')
                            # if '"horizon":192' in line:
                            #     line = line.replace('"horizon": 96', '"horizon": 192')
                            # elif '"horizon":336' in line:
                            #     line = line.replace('"horizon": 96', '"horizon": 336')
                            # elif '"horizon":720' in line:
                            #     line = line.replace('"horizon": 96', '"horizon": 720')
                                
                            # if '"is_train": 0' in line:
                            #     line = line.replace('"is_train": 0', '"is_train": 1')
                            # if ', "get_train": 1' in line:
                            #     line = line.replace(', "get_train": 1', '')
                        # line = line.replace('run_benchmark', 'run')
                        # line = line.replace('--save-path "', '--save-path "FULL/')
                        # line = line.replace('"freq": "h"', '"freq": "min"')
                        # line = line.replace('"freq": "h"', '"freq": "d"')
                        # if '"enc_in": 7' in line:
                        #     line = line.replace('"enc_in": 7', '"enc_in": 11')
                        # line = line.replace('FEW', 'FULL')
                        # # line = line.replace('}\'  --gpus', ', "sampling_rate": 0.05}\' --gpus')
                        # if "TimesFM" in line:
                        #     line = line.replace('TimesFM_adapter', 'PreTrain_adapter')
                            # if "ETTh1" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 64}\' --adapter')
                            # if "ETTh2" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 64}\' --adapter')
                            # if "ETTm1" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 128}\' --adapter')
                            # if "ETTm2" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 128}\' --adapter')
                            # if "Electricity" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')
                            # if "Weather" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 128}\' --adapter')
                            # if "Traffic" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 64}\' --adapter')
                            # if "Exchange" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')
                            # if "ZafNoo" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 64}\' --adapter')
                            # if "Solar" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')
                            # if "ILI" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')
                            # if "NN5" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')
                            # if "Wike2000" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')
                            # if "NASDAQ" in line:
                            #     line = line.replace('}\' --adapter', '"patch_size": 32}\' --adapter')    
                            
                        # if "UniTime" in line:
                        #     if '"seq_len": 96' in line:
                        #         line = line.replace('"max_token_num": 32, "max_backcast_len": 336', '"max_token_num": 17, "max_backcast_len": 96')
                        #         line = line.replace('"max_token_num": 43, "max_backcast_len": 512', '"max_token_num": 17, "max_backcast_len": 96')
                        #     if '"seq_len": 336' in line:
                        #         line = line.replace('"max_token_num": 17, "max_backcast_len": 96', '"max_token_num": 32, "max_backcast_len": 336')
                        #         line = line.replace('"max_token_num": 43, "max_backcast_len": 512', '"max_token_num": 32, "max_backcast_len": 336')
                        #     if '"seq_len": 512' in line:
                        #         line = line.replace('"max_token_num": 32, "max_backcast_len": 336', '"max_token_num": 43, "max_backcast_len": 512')
                        #         line = line.replace('"max_token_num": 17, "max_backcast_len": 96', '"max_token_num": 43, "max_backcast_len": 512')
                        
                        # if "ILI" in line or "NASDAQ" in line or "NN5" in line or "Wike2000" in line:
                            # line = line.replace('"horizon":96', '"horizon": 24')  
                            # line = line.replace('"horizon":192', '"horizon": 36')  
                            # line = line.replace('"horizon":336', '"horizon": 48')  
                            # line = line.replace('"horizon":720', '"horizon": 60')  
                            # line = line.replace('"horizon": 96', '"horizon": 24')  
                            # line = line.replace('"horizon": 192', '"horizon": 36')  
                            # line = line.replace('"horizon": 336', '"horizon": 48')  
                            # line = line.replace('"horizon": 720', '"horizon": 60')  
                            
                            # if "Timer" in line:
                            #     line = line.replace('"horizon": 24, "seq_len": 96', '"horizon": 96, "seq_len": 96')
                            #     line = line.replace('"horizon": 36, "seq_len": 96', '"horizon": 96, "seq_len": 96')
                            #     line = line.replace('"horizon": 48, "seq_len": 96', '"horizon": 96, "seq_len": 96')
                            #     line = line.replace('"horizon": 60, "seq_len": 96', '"horizon": 96, "seq_len": 96')
                            # elif "TimesFM" in line:
                            #     line = line.replace('"seq_len": 320', '"seq_len": 32, "label_len": 16')
                            # # elif "UniTS" in line:
                            # #     line = line.replace('"seq_len": 96', '"seq_len": 36, "label_len": 18')
                            # #     line = line.replace('"seq_len": 336', '"seq_len": 104, "label_len": 52')
                            
                            # if "TinyTimeMixer" in line:
                            #     line = line.replace(', "is_train": 0', ', "is_train": 1, "sampling_rate": 1')
                            # else:
                            #     line = line.replace('"seq_len": 96', '"seq_len": 36, "label_len": 18')
                            #     line = line.replace('"seq_len": 336', '"seq_len": 104, "label_len": 52')
                        f.write(line)

# 调用函数，传入你想要遍历的目录路径
update_sh_files("/root/FM4TS-Bench/script")