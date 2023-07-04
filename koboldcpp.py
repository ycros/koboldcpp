# A hacky little script from Concedo that exposes llama.cpp function bindings
# allowing it to be used via a simulated kobold api endpoint
# generation delay scales linearly with original prompt length.

import ctypes
import os
import argparse
import json, sys, http.server, time, asyncio, socket, threading
from concurrent.futures import ThreadPoolExecutor

stop_token_max = 10
sampler_order_max = 7

class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("batch_size", ctypes.c_int),
                ("f16_kv", ctypes.c_bool),
                ("low_vram", ctypes.c_bool),
                ("executable_path", ctypes.c_char_p),
                ("model_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("lora_base", ctypes.c_char_p),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("use_smartcontext", ctypes.c_bool),
                ("unban_tokens", ctypes.c_bool),
                ("clblast_info", ctypes.c_int),
                ("blasbatchsize", ctypes.c_int),
                ("debugmode", ctypes.c_int),
                ("forceversion", ctypes.c_int),
                ("gpulayers", ctypes.c_int)]

class generation_inputs(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("max_context_length", ctypes.c_int),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_a", ctypes.c_float),
                ("top_p", ctypes.c_float),
                ("typical_p", ctypes.c_float),
                ("tfs", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int),
                ("mirostat", ctypes.c_int),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("sampler_order", ctypes.c_int * sampler_order_max),
                ("sampler_len", ctypes.c_int),
                ("stop_sequence", ctypes.c_char_p * stop_token_max),
                ("stream_sse", ctypes.c_bool)]

class generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("text", ctypes.c_char * 16384)]

handle = None

def getdirpath():
    return os.path.dirname(os.path.realpath(__file__))
def file_exists(filename):
    return os.path.exists(os.path.join(getdirpath(), filename))

def pick_existant_file(ntoption,nonntoption):
    ntexist = file_exists(ntoption)
    nonntexist = file_exists(nonntoption)
    if os.name == 'nt':
        if nonntexist and not ntexist:
            return nonntoption
        return ntoption
    else:
        if ntexist and not nonntexist:
            return ntoption
        return nonntoption

lib_default = pick_existant_file("koboldcpp.dll","koboldcpp.so")
lib_failsafe = pick_existant_file("koboldcpp_failsafe.dll","koboldcpp_failsafe.so")
lib_openblas = pick_existant_file("koboldcpp_openblas.dll","koboldcpp_openblas.so")
lib_openblas_noavx2 = pick_existant_file("koboldcpp_openblas_noavx2.dll","koboldcpp_openblas_noavx2.so")
lib_clblast = pick_existant_file("koboldcpp_clblast.dll","koboldcpp_clblast.so")
lib_cublas = pick_existant_file("koboldcpp_cublas.dll","koboldcpp_cublas.so")


def init_library():
    global handle
    global lib_default,lib_failsafe,lib_openblas,lib_openblas_noavx2,lib_clblast,lib_cublas

    libname = ""
    use_blas = False # if true, uses OpenBLAS for acceleration. libopenblas.dll must exist in the same dir.
    use_clblast = False #uses CLBlast instead
    use_cublas = False #uses cublas instead
    use_noavx2 = False #uses openblas with no avx2 instructions
    if args.noavx2:
        use_noavx2 = True
        if not file_exists(lib_openblas_noavx2) or (os.name=='nt' and not file_exists("libopenblas.dll")):
            print("Warning: OpenBLAS library file not found. Non-BLAS library will be used.")
        elif args.noblas:
            print("!!! Attempting to use FAILSAFE MODE !!!")
        else:
            use_blas = True
            print("Attempting to use non-avx2 compatibility library with OpenBLAS. A compatible libopenblas will be required.")
    elif args.useclblast:
        if not file_exists(lib_clblast) or (os.name=='nt' and not file_exists("clblast.dll")):
            print("Warning: CLBlast library file not found. Non-BLAS library will be used.")
        else:
            print("Attempting to use CLBlast library for faster prompt ingestion. A compatible clblast will be required.")
            use_clblast = True
    elif (args.usecublas and args.usecublas!=""):
        if not file_exists(lib_cublas):
            print("Warning: CuBLAS library file not found. Non-BLAS library will be used.")
        else:
            print("Attempting to use CuBLAS library for faster prompt ingestion. A compatible CuBLAS will be required.")
            use_cublas = True
    else:
        if not file_exists(lib_openblas) or (os.name=='nt' and not file_exists("libopenblas.dll")):
            print("Warning: OpenBLAS library file not found. Non-BLAS library will be used.")
        elif args.noblas:
            print("Attempting to library without OpenBLAS.")
        else:
            use_blas = True
            print("Attempting to use OpenBLAS library for faster prompt ingestion. A compatible libopenblas will be required.")
            if sys.platform=="darwin":
                print("Mac OSX note: Some people have found Accelerate actually faster than OpenBLAS. To compare, run Koboldcpp with --noblas instead.")

    if use_noavx2:
        if use_blas:
            libname = lib_openblas_noavx2
        else:
            libname = lib_failsafe
    else:
        if use_clblast:
            libname = lib_clblast
        elif use_cublas:
            libname = lib_cublas
        elif use_blas:
            libname = lib_openblas
        else:
            libname = lib_default

    print("Initializing dynamic library: " + libname)
    dir_path = getdirpath()

    #OpenBLAS should provide about a 2x speedup on prompt ingestion if compatible.
    handle = ctypes.CDLL(os.path.join(dir_path, libname))

    handle.load_model.argtypes = [load_model_inputs]
    handle.load_model.restype = ctypes.c_bool
    handle.generate.argtypes = [generation_inputs, ctypes.c_wchar_p] #apparently needed for osx to work. i duno why they need to interpret it that way but whatever
    handle.generate.restype = generation_outputs
    handle.new_token.restype = ctypes.c_char_p
    handle.new_token.argtypes = [ctypes.c_int]
    handle.get_stream_count.restype = ctypes.c_int
    handle.has_finished.restype = ctypes.c_bool
    handle.abort_generate.restype = ctypes.c_bool
    handle.get_pending_output.restype = ctypes.c_char_p

def load_model(model_filename):
    inputs = load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.batch_size = 8
    inputs.max_context_length = maxctx #initial value to use for ctx, can be overwritten
    inputs.threads = args.threads
    inputs.low_vram = (True if args.usecublas=="lowvram" else False)
    inputs.blasthreads = args.blasthreads
    inputs.f16_kv = True
    inputs.use_mmap = (not args.nommap)
    inputs.use_mlock = args.usemlock
    inputs.lora_filename = "".encode("UTF-8")
    inputs.lora_base = "".encode("UTF-8")
    if args.lora:
        inputs.lora_filename = args.lora[0].encode("UTF-8")
        inputs.use_mmap = False
        if len(args.lora) > 1:
            inputs.lora_base = args.lora[1].encode("UTF-8")
    inputs.use_smartcontext = args.smartcontext
    inputs.unban_tokens = args.unbantokens
    inputs.blasbatchsize = args.blasbatchsize
    inputs.forceversion = args.forceversion
    inputs.gpulayers = args.gpulayers
    clblastids = 0
    if args.useclblast:
        clblastids = 100 + int(args.useclblast[0])*10 + int(args.useclblast[1])
    inputs.clblast_info = clblastids
    inputs.executable_path = (getdirpath()+"/").encode("UTF-8")
    inputs.debugmode = args.debugmode
    ret = handle.load_model(inputs)
    return ret

def generate(prompt,max_length=20, max_context_length=512, temperature=0.8, top_k=120, top_a=0.0, top_p=0.85, typical_p=1.0, tfs=1.0, rep_pen=1.1, rep_pen_range=128, mirostat=0, mirostat_tau=5.0, mirostat_eta=0.1, sampler_order=None, seed=-1, stop_sequence=[], stream_sse=False):
    inputs = generation_inputs()
    outputs = ctypes.create_unicode_buffer(ctypes.sizeof(generation_outputs))
    inputs.prompt = prompt.encode("UTF-8")
    inputs.max_context_length = max_context_length   # this will resize the context buffer if changed
    inputs.max_length = max_length
    inputs.temperature = temperature
    inputs.top_k = top_k
    inputs.top_a = top_a
    inputs.top_p = top_p
    inputs.typical_p = typical_p
    inputs.tfs = tfs
    inputs.rep_pen = rep_pen
    inputs.rep_pen_range = rep_pen_range
    inputs.stream_sse = stream_sse
    if args.usemirostat and args.usemirostat[0]>0:
        inputs.mirostat = int(args.usemirostat[0])
        inputs.mirostat_tau = float(args.usemirostat[1])
        inputs.mirostat_eta = float(args.usemirostat[2])
    elif mirostat in (1, 2):
        inputs.mirostat = mirostat
        inputs.mirostat_tau = mirostat_tau
        inputs.mirostat_eta = mirostat_eta
    else:
        inputs.mirostat = inputs.mirostat_tau = inputs.mirostat_eta = 0
    if sampler_order and 0 < len(sampler_order) <= sampler_order_max:
        try:
            for i, sampler in enumerate(sampler_order):
                inputs.sampler_order[i] = sampler
            inputs.sampler_len = len(sampler_order)
        except TypeError as e:
            print("ERROR: sampler_order must be a list of integers: " + str(e))
    inputs.seed = seed
    for n in range(stop_token_max):
        if not stop_sequence or n >= len(stop_sequence):
            inputs.stop_sequence[n] = "".encode("UTF-8")
        else:
            inputs.stop_sequence[n] = stop_sequence[n].encode("UTF-8")
    ret = handle.generate(inputs,outputs)
    if(ret.status==1):
        return ret.text.decode("UTF-8","ignore")
    return ""

def utfprint(str):
    try:
        print(str)
    except UnicodeEncodeError:
        # Replace or omit the problematic character
        utf_string = str.encode('ascii', 'ignore').decode('ascii')
        print(utf_string)

#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
friendlymodelname = "concedo/koboldcpp"  # local kobold api apparently needs a hardcoded known HF model name
maxctx = 2048
maxhordectx = 1024
maxhordelen = 256
modelbusy = False
defaultport = 5001
KcppVersion = "1.34"
showdebug = True

class ServerRequestHandler(http.server.SimpleHTTPRequestHandler):
    sys_version = ""
    server_version = "ConcedoLlamaForKoboldServer"

    def __init__(self, addr, port, embedded_kailite):
        self.addr = addr
        self.port = port
        self.embedded_kailite = embedded_kailite

    def __call__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        global showdebug
        if showdebug:
            super().log_message(format, *args)
        pass

    async def generate_text(self, newprompt, genparams, basic_api_flag, stream_flag):

        def run_blocking():
            if basic_api_flag:
                return generate(
                    prompt=newprompt,
                    max_length=genparams.get('max', 50),
                    temperature=genparams.get('temperature', 0.8),
                    top_k=int(genparams.get('top_k', 120)),
                    top_a=genparams.get('top_a', 0.0),
                    top_p=genparams.get('top_p', 0.85),
                    typical_p=genparams.get('typical', 1.0),
                    tfs=genparams.get('tfs', 1.0),
                    rep_pen=genparams.get('rep_pen', 1.1),
                    rep_pen_range=genparams.get('rep_pen_range', 128),
                    mirostat=genparams.get('mirostat', 0),
                    mirostat_tau=genparams.get('mirostat_tau', 5.0),
                    mirostat_eta=genparams.get('mirostat_eta', 0.1),
                    sampler_order=genparams.get('sampler_order', None),
                    seed=genparams.get('sampler_seed', -1),
                    stop_sequence=genparams.get('stop_sequence', []),
                    stream_sse=stream_flag)

            else:
                return generate(prompt=newprompt,
                    max_context_length=genparams.get('max_context_length', maxctx),
                    max_length=genparams.get('max_length', 50),
                    temperature=genparams.get('temperature', 0.8),
                    top_k=genparams.get('top_k', 120),
                    top_a=genparams.get('top_a', 0.0),
                    top_p=genparams.get('top_p', 0.85),
                    typical_p=genparams.get('typical', 1.0),
                    tfs=genparams.get('tfs', 1.0),
                    rep_pen=genparams.get('rep_pen', 1.1),
                    rep_pen_range=genparams.get('rep_pen_range', 128),
                    mirostat=genparams.get('mirostat', 0),
                    mirostat_tau=genparams.get('mirostat_tau', 5.0),
                    mirostat_eta=genparams.get('mirostat_eta', 0.1),
                    sampler_order=genparams.get('sampler_order', None),
                    seed=genparams.get('sampler_seed', -1),
                    stop_sequence=genparams.get('stop_sequence', []),
                    stream_sse=stream_flag)

        recvtxt = ""
        if stream_flag:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor()
            recvtxt = await loop.run_in_executor(executor, run_blocking)
        else:
            recvtxt = run_blocking()

        if args.debugmode!=-1:
            utfprint("\nOutput: " + recvtxt)

        res = {"data": {"seqs":[recvtxt]}} if basic_api_flag else {"results": [{"text": recvtxt}]}

        try:
            return res
        except Exception as e:
            print(f"Generate: Error while generating: {e}")


    async def send_sse_event(self, event, data):
        self.wfile.write(f'event: {event}\n'.encode())
        self.wfile.write(f'data: {data}\n\n'.encode())


    async def handle_sse_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        current_token = 0

        while not handle.has_finished():
            if current_token < handle.get_stream_count():
                token = handle.new_token(current_token)

                if token is None: # Token isnt ready yet, received nullpointer
                    continue

                current_token += 1

                tokenStr = ctypes.string_at(token).decode("UTF-8","ignore")
                event_data = {"token": tokenStr}
                event_str = json.dumps(event_data)
                await self.send_sse_event("message", event_str)

            await asyncio.sleep(0)

        # flush buffers, sleep a bit to make sure all data sent, and then force close the connection
        self.wfile.flush()
        await asyncio.sleep(0.1)
        self.close_connection = True


    async def handle_request(self, genparams, newprompt, basic_api_flag, stream_flag):
        tasks = []

        if stream_flag:
            tasks.append(self.handle_sse_stream())

        generate_task = asyncio.create_task(self.generate_text(newprompt, genparams, basic_api_flag, stream_flag))
        tasks.append(generate_task)

        try:
            await asyncio.gather(*tasks)
            generate_result = generate_task.result()
            return generate_result
        except Exception as e:
            print(e)


    def do_GET(self):
        global maxctx, maxhordelen, friendlymodelname, KcppVersion, streamLock
        self.path = self.path.rstrip('/')
        response_body = None

        if self.path in ["", "/?"] or self.path.startswith(('/?','?')): #it's possible for the root url to have ?params without /
            if args.stream and not "streaming=1" in self.path:
                self.path = self.path.replace("streaming=0","")
                if self.path.startswith(('/?','?')):
                    self.path += "&streaming=1"
                else:
                    self.path = self.path + "?streaming=1"
                self.send_response(302)
                self.send_header("Location", self.path)
                self.end_headers()
                print("Force redirect to streaming mode, as --stream is set.")
                return None

            if self.embedded_kailite is None:
                response_body = (f"Embedded Kobold Lite is not found.<br>You will have to connect via the main KoboldAI client, or <a href='https://lite.koboldai.net?local=1&port={self.port}'>use this URL</a> to connect.").encode()
            else:
                response_body = self.embedded_kailite

        elif self.path.endswith(('/api/v1/model', '/api/latest/model')):
            response_body = (json.dumps({'result': friendlymodelname }).encode())

        elif self.path.endswith(('/api/v1/config/max_length', '/api/latest/config/max_length')):
            response_body = (json.dumps({"value": maxhordelen}).encode())

        elif self.path.endswith(('/api/v1/config/max_context_length', '/api/latest/config/max_context_length')):
            response_body = (json.dumps({"value": min(maxctx,maxhordectx)}).encode())

        elif self.path.endswith(('/api/v1/config/soft_prompt', '/api/latest/config/soft_prompt')):
            response_body = (json.dumps({"value":""}).encode())

        elif self.path.endswith(('/api/v1/config/soft_prompts_list', '/api/latest/config/soft_prompts_list')):
            response_body = (json.dumps({"values": []}).encode())

        elif self.path.endswith(('/api/v1/info/version', '/api/latest/info/version')):
            response_body = (json.dumps({"result":"1.2.2"}).encode())

        elif self.path.endswith(('/api/extra/version')):
            response_body = (json.dumps({"result":"KoboldCpp","version":KcppVersion}).encode())

        if response_body is None:
            self.send_response(404)
            self.end_headers()
            rp = 'Error: HTTP Server is running, but this endpoint does not exist. Please check the URL.'
            self.wfile.write(rp.encode())
        else:
            self.send_response(200)
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        return

    def do_POST(self):
        global modelbusy
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        basic_api_flag = False
        kai_api_flag = False
        kai_sse_stream_flag = False
        self.path = self.path.rstrip('/')

        if self.path.endswith('/api/extra/abort'):
            ag = handle.abort_generate()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"success": ("true" if ag else "false")}).encode())
            print("\nGeneration Aborted")
            modelbusy = False
            return

        if self.path.endswith('/api/extra/generate/check'):
            pendtxt = handle.get_pending_output()
            pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"results": [{"text": pendtxtStr}]}).encode())
            return

        if modelbusy:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(json.dumps({"detail": {
                    "msg": "Server is busy; please try again later.",
                    "type": "service_unavailable",
                }}).encode())
            return

        if self.path.endswith('/request'):
            basic_api_flag = True

        if self.path.endswith(('/api/v1/generate', '/api/latest/generate')):
            kai_api_flag = True

        if self.path.endswith('/api/extra/generate/stream'):
            kai_api_flag = True
            kai_sse_stream_flag = True

        if basic_api_flag or kai_api_flag:
            genparams = None
            try:
                genparams = json.loads(body)
            except ValueError as e:
                utfprint("Body Err: " + str(body))
                return self.send_response(503)

            if args.debugmode!=-1:
                utfprint("\nInput: " + json.dumps(genparams))

            modelbusy = True

            if kai_api_flag:
                fullprompt = genparams.get('prompt', "")
            else:
                fullprompt = genparams.get('text', "")
            newprompt = fullprompt

            gen = asyncio.run(self.handle_request(genparams, newprompt, basic_api_flag, kai_sse_stream_flag))

            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(gen).encode())
            except:
                print("Generate: The response could not be sent, maybe connection was terminated?")

            modelbusy = False

            return

        self.send_response(404)
        self.end_headers()


    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        if "/api" in self.path:
            if self.path.endswith("/stream"):
                self.send_header('Content-type', 'text/event-stream')
            self.send_header('Content-type', 'application/json')
        else:
            self.send_header('Content-type', 'text/html')
        return super(ServerRequestHandler, self).end_headers()


def RunServerMultiThreaded(addr, port, embedded_kailite = None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((addr, port))
    sock.listen(5)

    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()

        def run(self):
            handler = ServerRequestHandler(addr, port, embedded_kailite)
            with http.server.HTTPServer((addr, port), handler, False) as self.httpd:
                try:
                    self.httpd.socket = sock
                    self.httpd.server_bind = self.server_close = lambda self: None
                    self.httpd.serve_forever()
                except (KeyboardInterrupt,SystemExit):
                    self.httpd.server_close()
                    sys.exit(0)
                finally:
                    self.httpd.server_close()
                    sys.exit(0)
        def stop(self):
            self.httpd.server_close()

    numThreads = 6
    threadArr = []
    for i in range(numThreads):
        threadArr.append(Thread(i))
    while 1:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            for i in range(numThreads):
                threadArr[i].stop()
            sys.exit(0)


def show_gui():
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

    if len(sys.argv) == 1:
        #no args passed at all. Show nooby gui
        root = tk.Tk()
        launchclicked = False

        def guilaunch():
            nonlocal launchclicked
            launchclicked = True
            root.destroy()
            pass

        # Adjust size
        root.geometry("480x360")
        root.title("KoboldCpp v"+KcppVersion)
        root.grid_columnconfigure(0, weight=1)
        tk.Label(root, text = "KoboldCpp Easy Launcher",
                font = ("Arial", 12)).grid(row=0,column=0)
        tk.Label(root, text = "(Note: KoboldCpp only works with GGML model formats!)",
                font = ("Arial", 9)).grid(row=1,column=0)

        blasbatchopts = ["Don't Batch BLAS","BLAS = 32","BLAS = 64","BLAS = 128","BLAS = 256","BLAS = 512","BLAS = 1024"]
        blaschoice = tk.StringVar()
        blaschoice.set("BLAS = 512")

        runopts = ["Use OpenBLAS","Use CLBLast GPU #1","Use CLBLast GPU #2","Use CLBLast GPU #3","Use CuBLAS GPU","Use No BLAS","Use OpenBLAS (Old CPU, noavx2)","Failsafe Mode (Old CPU, noavx)"]
        runchoice = tk.StringVar()
        runchoice.set("Use OpenBLAS")

        def onDropdownChange(event):
            sel = runchoice.get()
            if sel==runopts[1] or sel==runopts[2] or sel==runopts[3] or sel==runopts[4]:
                frameC.grid(row=4,column=0,pady=4)
            else:
                frameC.grid_forget()

        frameA = tk.Frame(root)
        tk.OptionMenu( frameA , runchoice , command = onDropdownChange ,*runopts ).grid(row=0,column=0)
        tk.OptionMenu( frameA , blaschoice ,*blasbatchopts ).grid(row=0,column=1)
        frameA.grid(row=2,column=0)

        frameB = tk.Frame(root)
        threads_var=tk.StringVar()
        threads_var.set(str(default_threads))
        threads_lbl = tk.Label(frameB, text = 'Threads: ', font=('calibre',10, 'bold'))
        threads_input = tk.Entry(frameB,textvariable = threads_var, font=('calibre',10,'normal'))
        threads_lbl.grid(row=0,column=0)
        threads_input.grid(row=0,column=1)
        frameB.grid(row=3,column=0,pady=4)

        frameC = tk.Frame(root)
        gpu_layers_var=tk.StringVar()
        gpu_layers_var.set("0")
        gpu_lbl = tk.Label(frameC, text = 'GPU Layers: ', font=('calibre',10, 'bold'))
        gpu_layers_input = tk.Entry(frameC,textvariable = gpu_layers_var, font=('calibre',10,'normal'))
        gpu_lbl.grid(row=0,column=0)
        gpu_layers_input.grid(row=0,column=1)
        frameC.grid(row=4,column=0,pady=4)
        onDropdownChange(None)

        stream = tk.IntVar()
        smartcontext = tk.IntVar()
        launchbrowser = tk.IntVar(value=1)
        unbantokens = tk.IntVar()
        highpriority = tk.IntVar()
        disablemmap = tk.IntVar()
        frameD = tk.Frame(root)
        tk.Checkbutton(frameD, text='Streaming Mode',variable=stream, onvalue=1, offvalue=0).grid(row=0,column=0)
        tk.Checkbutton(frameD, text='Use SmartContext',variable=smartcontext, onvalue=1, offvalue=0).grid(row=0,column=1)
        tk.Checkbutton(frameD, text='High Priority',variable=highpriority, onvalue=1, offvalue=0).grid(row=1,column=0)
        tk.Checkbutton(frameD, text='Disable MMAP',variable=disablemmap, onvalue=1, offvalue=0).grid(row=1,column=1)
        tk.Checkbutton(frameD, text='Unban Tokens',variable=unbantokens, onvalue=1, offvalue=0).grid(row=2,column=0)
        tk.Checkbutton(frameD, text='Launch Browser',variable=launchbrowser, onvalue=1, offvalue=0).grid(row=2,column=1)
        frameD.grid(row=5,column=0,pady=4)

        # Create button, it will change label text
        tk.Button( root , text = "Launch", font = ("Impact", 18), bg='#54FA9B', command = guilaunch ).grid(row=6,column=0)
        tk.Label(root, text = "(Please use the Command Line for more advanced options)",
                font = ("Arial", 9)).grid(row=7,column=0)

        root.mainloop()

        if launchclicked==False:
            print("Exiting by user request.")
            time.sleep(2)
            sys.exit()

        #load all the vars
        args.threads = int(threads_var.get())
        args.gpulayers = int(gpu_layers_var.get())

        args.stream = (stream.get()==1)
        args.smartcontext = (smartcontext.get()==1)
        args.launch = (launchbrowser.get()==1)
        args.unbantokens = (unbantokens.get()==1)
        args.highpriority = (highpriority.get()==1)
        args.nommap = (disablemmap.get()==1)
        selrunchoice = runchoice.get()
        selblaschoice = blaschoice.get()

        if selrunchoice==runopts[1]:
            args.useclblast = [0,0]
        if selrunchoice==runopts[2]:
            args.useclblast = [1,0]
        if selrunchoice==runopts[3]:
            args.useclblast = [0,1]
        if selrunchoice==runopts[4]:
            args.usecublas = True
        if selrunchoice==runopts[5]:
            args.noblas = True
        if selrunchoice==runopts[6]:
            args.noavx2 = True
        if selrunchoice==runopts[7]:
            args.noavx2 = True
            args.noblas = True
            args.nommap = True
            print("[Failsafe Mode : mmap is disabled.]")

        if selblaschoice==blasbatchopts[0]:
            args.blasbatchsize = -1
        if selblaschoice==blasbatchopts[1]:
            args.blasbatchsize = 32
        if selblaschoice==blasbatchopts[2]:
            args.blasbatchsize = 64
        if selblaschoice==blasbatchopts[3]:
            args.blasbatchsize = 128
        if selblaschoice==blasbatchopts[4]:
            args.blasbatchsize = 256
        if selblaschoice==blasbatchopts[5]:
            args.blasbatchsize = 512
        if selblaschoice==blasbatchopts[6]:
            args.blasbatchsize = 1024

        root = tk.Tk()
        root.attributes("-alpha", 0)
        args.model_param = askopenfilename(title="Select ggml model .bin files")
        root.destroy()
        if not args.model_param:
            print("\nNo ggml model file was selected. Exiting.")
            time.sleep(2)
            sys.exit(2)

    else:
        root = tk.Tk() #we dont want the useless window to be visible, but we want it in taskbar
        root.attributes("-alpha", 0)
        args.model_param = askopenfilename(title="Select ggml model .bin files")
        root.destroy()
        if not args.model_param:
            print("\nNo ggml model file was selected. Exiting.")
            time.sleep(2)
            sys.exit(2)

def main(args):

    embedded_kailite = None
    if not args.model_param:
        args.model_param = args.model
    if not args.model_param:
        #give them a chance to pick a file
        print("For command line arguments, please refer to --help")
        print("Otherwise, please manually select ggml file:")
        try:
            show_gui()
        except Exception as ex:
            print("File selection GUI unsupported. Please check command line: script.py --help")
            print("Reason for no GUI: " + str(ex))
            time.sleep(2)
            sys.exit(2)

    if args.hordeconfig and args.hordeconfig[0]!="":
        global friendlymodelname, maxhordelen, maxhordectx, showdebug
        friendlymodelname = "koboldcpp/"+args.hordeconfig[0]
        if len(args.hordeconfig) > 1:
            maxhordelen = int(args.hordeconfig[1])
        if len(args.hordeconfig) > 2:
            maxhordectx = int(args.hordeconfig[2])
        if args.debugmode == 0:
            args.debugmode = -1

    if args.debugmode != 1:
        showdebug = False

    if args.highpriority:
        print("Setting process to Higher Priority - Use Caution")
        try:
            import psutil
            os_used = sys.platform
            process = psutil.Process(os.getpid())  # Set high priority for the python script for the CPU
            oldprio = process.nice()
            if os_used == "win32":  # Windows (either 32-bit or 64-bit)
                process.nice(psutil.REALTIME_PRIORITY_CLASS)
                print("High Priority for Windows Set: " + str(oldprio) + " to " + str(process.nice()))
            elif os_used == "linux":  # linux
                process.nice(psutil.IOPRIO_CLASS_RT)
                print("High Priority for Linux Set: " + str(oldprio) + " to " + str(process.nice()))
            else:  # MAC OS X or other
                process.nice(-18)
                print("High Priority for Other OS Set :" + str(oldprio) + " to " + str(process.nice()))
        except Exception as ex:
             print("Error, Could not change process priority: " + str(ex))

    if args.contextsize:
        global maxctx
        maxctx = args.contextsize

    init_library() # Note: if blas does not exist and is enabled, program will crash.
    print("==========")
    time.sleep(1)
    if not os.path.exists(args.model_param):
        print(f"Cannot find model file: {args.model_param}")
        time.sleep(2)
        sys.exit(2)

    if args.lora and args.lora[0]!="":
        if not os.path.exists(args.lora[0]):
            print(f"Cannot find lora file: {args.lora[0]}")
            time.sleep(2)
            sys.exit(2)
        else:
            args.lora[0] = os.path.abspath(args.lora[0])
            if len(args.lora) > 1:
                if not os.path.exists(args.lora[1]):
                    print(f"Cannot find lora base: {args.lora[1]}")
                    time.sleep(2)
                    sys.exit(2)
                else:
                    args.lora[1] = os.path.abspath(args.lora[1])

    if args.psutil_set_threads:
        import psutil
        args.threads = psutil.cpu_count(logical=False)
        print("Overriding thread count, using " + str(args.threads) + " threads instead.")

    if not args.blasthreads or args.blasthreads <= 0:
        args.blasthreads = args.threads

    modelname = os.path.abspath(args.model_param)
    print(f"Loading model: {modelname} \n[Threads: {args.threads}, BlasThreads: {args.blasthreads}, SmartContext: {args.smartcontext}]")
    loadok = load_model(modelname)
    print("Load Model OK: " + str(loadok))

    if not loadok:
        print("Could not load model: " + modelname)
        time.sleep(2)
        sys.exit(3)
    try:
        basepath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(basepath, "klite.embd"), mode='rb') as f:
            embedded_kailite = f.read()
            print("Embedded Kobold Lite loaded.")
    except:
        print("Could not find Kobold Lite. Embedded Kobold Lite will not be available.")

    if args.port_param!=defaultport:
        args.port = args.port_param
    print(f"Starting Kobold HTTP Server on port {args.port}")
    epurl = ""
    if args.host=="":
        epurl = f"http://localhost:{args.port}"
    else:
        epurl = f"http://{args.host}:{args.port}"

    if args.launch:
        try:
            import webbrowser as wb
            wb.open(epurl)
        except:
            print("--launch was set, but could not launch web browser automatically.")
    print(f"Please connect to custom endpoint at {epurl}")
    asyncio.run(RunServerMultiThreaded(args.host, args.port, embedded_kailite))

if __name__ == '__main__':
    print("Welcome to KoboldCpp - Version " + KcppVersion) # just update version manually
    # print("Python version: " + sys.version)
    parser = argparse.ArgumentParser(description='KoboldCpp Server')
    modelgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    modelgroup.add_argument("--model", help="Model file to load", nargs="?")
    modelgroup.add_argument("model_param", help="Model file to load (positional)", nargs="?")
    portgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    portgroup.add_argument("--port", help="Port to listen on", default=defaultport, type=int, action='store')
    portgroup.add_argument("port_param", help="Port to listen on (positional)", default=defaultport, nargs="?", type=int, action='store')
    parser.add_argument("--host", help="Host IP to listen on. If empty, all routable interfaces are accepted.", default="")
    parser.add_argument("--launch", help="Launches a web browser when load is completed.", action='store_true')
    parser.add_argument("--lora", help="LLAMA models only, applies a lora file on top of model. Experimental.", metavar=('[lora_filename]', '[lora_base]'), nargs='+')
    physical_core_limit = 1
    if os.cpu_count()!=None and os.cpu_count()>1:
        physical_core_limit = int(os.cpu_count()/2)
    default_threads = (physical_core_limit if physical_core_limit<=3 else max(3,physical_core_limit-1))
    parser.add_argument("--threads", help="Use a custom number of threads if specified. Otherwise, uses an amount based on CPU cores", type=int, default=default_threads)
    parser.add_argument("--blasthreads", help="Use a different number of threads during BLAS if specified. Otherwise, has the same value as --threads",metavar=('[threads]'), type=int, default=0)
    parser.add_argument("--psutil_set_threads", help="Experimental flag. If set, uses psutils to determine thread count based on physical cores.", action='store_true')
    parser.add_argument("--highpriority", help="Experimental flag. If set, increases the process CPU priority, potentially speeding up generation. Use caution.", action='store_true')
    parser.add_argument("--contextsize", help="Controls the memory allocated for maximum context size, only change if you need more RAM for big contexts. (default 2048)", type=int,choices=[512,1024,2048,3072,4096,6144,8192], default=2048)
    parser.add_argument("--blasbatchsize", help="Sets the batch size used in BLAS processing (default 512). Setting it to -1 disables BLAS mode, but keeps other benefits like GPU offload.", type=int,choices=[-1,32,64,128,256,512,1024], default=512)
    parser.add_argument("--stream", help="Uses streaming when generating tokens. Only for the Kobold Lite UI.", action='store_true')
    parser.add_argument("--smartcontext", help="Reserving a portion of context to try processing less frequently.", action='store_true')
    parser.add_argument("--unbantokens", help="Normally, KoboldAI prevents certain tokens such as EOS and Square Brackets. This flag unbans them.", action='store_true')
    parser.add_argument("--usemirostat", help="Experimental! Replaces your samplers with mirostat. Takes 3 params = [type(0/1/2), tau(5.0), eta(0.1)].",metavar=('[type]', '[tau]', '[eta]'), type=float, nargs=3)
    parser.add_argument("--forceversion", help="If the model file format detection fails (e.g. rogue modified model) you can set this to override the detected format (enter desired version, e.g. 401 for GPTNeoX-Type2).",metavar=('[version]'), type=int, default=0)
    parser.add_argument("--nommap", help="If set, do not use mmap to load newer models", action='store_true')
    parser.add_argument("--usemlock", help="For Apple Systems. Force system to keep model in RAM rather than swapping or compressing", action='store_true')
    parser.add_argument("--noavx2", help="Do not use AVX2 instructions, a slower compatibility mode for older devices. Does not work with --clblast.", action='store_true')
    parser.add_argument("--debugmode", help="Shows additional debug info in the terminal.", action='store_const', const=1, default=0)
    parser.add_argument("--skiplauncher", help="Doesn't display or use the new GUI launcher.", action='store_true')
    parser.add_argument("--hordeconfig", help="Sets the display model name to something else, for easy use on AI Horde. Optional additional parameters set the horde max genlength and max ctxlen.",metavar=('[hordename]', '[hordelength] [hordectx]'), nargs='+')
    compatgroup = parser.add_mutually_exclusive_group()
    compatgroup.add_argument("--noblas", help="Do not use OpenBLAS for accelerated prompt ingestion", action='store_true')
    compatgroup.add_argument("--useclblast", help="Use CLBlast for GPU Acceleration. Must specify exactly 2 arguments, platform ID and device ID (e.g. --useclblast 1 0).", type=int, choices=range(0,9), nargs=2)
    compatgroup.add_argument("--usecublas", help="Use CuBLAS for GPU Acceleration. Requires Nvidia GPU. Select lowvram to not allocate VRAM scratch buffer.", default='', const='normal', nargs='?', choices=['normal', 'lowvram'])
    parser.add_argument("--gpulayers", help="Set number of layers to offload to GPU when using GPU. Requires GPU.",metavar=('[GPU layers]'), type=int, default=0)
    args = parser.parse_args()
    main(args)
