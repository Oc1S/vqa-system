from django.http import HttpResponse, HttpResponseBadRequest
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import apis.vqa.predict as predict


def index(request):
    return HttpResponse("APIs")


@csrf_exempt
def vqa(request):
    question = request.POST.get("question")
    img = request.FILES.get("image")
    if img is None or question is None:
        return HttpResponseBadRequest("Params Error")
    print(img, question)
    res = predict.pred(img, question)
    return JsonResponse(res, safe=False)
