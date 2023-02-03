from __future__ import unicode_literals

from django.db import models


class Document(models.Model):
    request = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    ShowSequence = models.BooleanField(default=False)
    inference = models.CharField(max_length=255, blank=True, default="")
    uploaded_at = models.DateTimeField(auto_now_add=True)
